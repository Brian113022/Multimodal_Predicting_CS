"""Flask API and single-page UI for Late-Fusion prediction."""

import os
import secrets
from pathlib import Path
from typing import Callable

import io
import json
import tempfile

from flask import (
    Flask,
    Response,
    abort,
    Request as FlaskRequest,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .config import ALLOWED_UPLOAD_SUFFIXES, Config
from .dicom_classification import ClassificationRules
from .extractors.dicom import LUMBAR_SEGMENTS, dicom_features_events, segment_levels
from .extractors import (
    ExtractionExecutionError,
    ExtractionInputError,
    ExtractorUnavailableError,
    available_levels,
    extract_muscle_features_events,
    extract_radiomics_features_events,
    normalise_levels,
)
from .explain import (
    ExplanationError,
    ExplanationUnavailableError,
    ShapExplainer,
)
from .inference import (
    BundleValidationError,
    LateFusionPredictor,
    PredictionInputError,
)
from .radiomics_batch import bp as radiomics_batch_bp


MODALITY_ORDER = ("Clinical", "Muscle", "Radiomics")


class MedicalUploadRequest(FlaskRequest):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        if self.path == "/api/upload/dicom":
            # A case can contain thousands of small slices. The default
            # per-file in-memory spool would retain their combined size.
            return tempfile.TemporaryFile(mode="w+b")
        return super()._get_file_stream(total_content_length, content_type, filename, content_length)


def _ndjson(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + chr(10)


def _allowed_upload(filename: str) -> bool:
    lower_name = filename.lower()
    return any(lower_name.endswith(suffix) for suffix in ALLOWED_UPLOAD_SUFFIXES)


def create_app(config_object=Config) -> Flask:
    app = Flask(__name__)
    app.request_class = MedicalUploadRequest
    app.config.from_object(config_object)
    app.extensions["extraction_file_signer"] = URLSafeTimedSerializer(secrets.token_hex(32), salt="extraction-files")

    predictor = None
    model_load_error = None
    try:
        predictor = LateFusionPredictor(Path(app.config["MODEL_BUNDLE_PATH"]))
        app.logger.info(
            "Loaded Late-Fusion model bundle from %s",
            app.config["MODEL_BUNDLE_PATH"],
        )
    except BundleValidationError as exc:
        # The UI remains available so it can report why prediction is disabled.
        # No endpoint ever trains or substitutes a mock model.
        model_load_error = str(exc)
        app.logger.warning(model_load_error)

    app.extensions["late_fusion_predictor"] = predictor
    app.extensions["model_load_error"] = model_load_error
    explainer = (
        ShapExplainer(predictor, app.config["SHAP_BACKGROUND_DIR"])
        if predictor is not None
        else None
    )
    app.extensions["shap_explainer"] = explainer
    # The batch Radiomics page needs no model, so it works in degraded mode too.
    app.register_blueprint(radiomics_batch_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/extraction-files/<token>")
    def extraction_file(token):
        try:
            data = app.extensions["extraction_file_signer"].loads(token, max_age=24 * 60 * 60)
            root = _segmentation_root(data["source"])
            directory = Path(data["directory"]).resolve()
            path = (directory / data["filename"]).resolve()
            if root is None or directory == root or not directory.is_relative_to(root) or not path.is_relative_to(directory) or not path.is_file():
                abort(404)
        except (BadSignature, KeyError, TypeError, ValueError, OSError):
            abort(404)
        response = send_file(path, as_attachment=request.args.get("download") == "1", conditional=False)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/health")
    def health():
        if predictor is None:
            return jsonify({
                "status": "degraded",
                "model_loaded": False,
                "error": model_load_error,
            }), 503
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model": predictor.model_info(),
        })

    @app.get("/api/model-info")
    def model_info():
        if predictor is None:
            return _model_unavailable(model_load_error)
        return jsonify(predictor.model_info())

    @app.get("/api/schema/clinical")
    def clinical_schema():
        if predictor is None:
            return _model_unavailable(model_load_error)
        return jsonify({
            "fields": predictor.clinical_schema,
            "count": len(predictor.clinical_schema),
        })

    @app.post("/api/upload/muscle")
    def upload_muscle():
        # Muscle extraction takes many seconds, so it streams NDJSON progress
        # events instead of leaving the browser guessing.
        return _handle_extraction(
            modality="Muscle",
            extractor=lambda image, mask, expected: extract_muscle_features_events(
                image,
                mask,
                expected,
                command=app.config["MUSCLEMAP_COMMAND"],
                segmentation_command=app.config[
                    "MUSCLEMAP_SEGMENT_COMMAND"
                ],
                segmentation_region=app.config[
                    "MUSCLEMAP_SEGMENT_REGION"
                ],
                method=app.config["MUSCLEMAP_METHOD"],
                region=app.config["MUSCLEMAP_REGION"],
                components=app.config["MUSCLEMAP_COMPONENTS"],
                timeout_seconds=app.config["EXTRACTION_TIMEOUT_SECONDS"],
                python_executable=app.config["MUSCLEMAP_PYTHON"],
            ),
            predictor=predictor,
            mask_mode="never",
        )

    @app.get("/api/schema/dicom")
    def dicom_schema():
        return jsonify({
            "levels": [{"value": code, "label": code} for code in LUMBAR_SEGMENTS],
            "max_upload_bytes": app.config["MAX_CONTENT_LENGTH"],
            "max_files": app.config["MAX_DICOM_FILES"],
        })

    @app.post("/api/upload/dicom")
    def upload_dicom():
        if predictor is None:
            return _model_unavailable(model_load_error)
        request.max_form_parts = app.config["MAX_FORM_PARTS"]
        uploads = [upload for upload in request.files.getlist("files") if upload.filename]
        level = request.form.get("level", "")
        try:
            segment_levels(level)
            if not uploads:
                raise ExtractionInputError("Select DICOM files and a lumbar level.")
            if len(uploads) > app.config["MAX_DICOM_FILES"]:
                raise ExtractionInputError(f"This selection exceeds the {app.config['MAX_DICOM_FILES']} file limit.")
        except ExtractionInputError as exc:
            return jsonify({"error": "invalid_dicom_input", "message": str(exc)}), 422
        temporary = tempfile.TemporaryDirectory(prefix="dicom_web_")
        directory = Path(temporary.name)
        try:
            paths = []
            # Keep uploads on disk before returning the streaming response;
            # filenames/relative browser paths never determine server paths.
            for index, upload in enumerate(uploads):
                path = directory / f"upload_{index:06d}.dcm"
                upload.save(path)
                paths.append(path)
        except Exception:
            temporary.cleanup()
            raise
        config = app.config
        events = dicom_features_events(
            paths, level, directory,
            expected_features={modality: predictor.expected_features(modality) for modality in ("Muscle", "Radiomics")},
            converter_command=config["DCM2NIIX_COMMAND"],
            conversion_timeout=config["DICOM_CONVERSION_TIMEOUT_SECONDS"],
            archive_root=config["DICOM_SELECTED_SERIES_DIR"],
            rules=ClassificationRules(
                config["DICOM_T2_TR_MIN_MS"], config["DICOM_T2_TE_MIN_MS"],
                config["DICOM_T1_TR_MAX_MS"], config["DICOM_T1_TE_MAX_MS"],
                t1_flair_tr_max=config["DICOM_T1_FLAIR_TR_MAX_MS"],
                t1_flair_ti_min=config["DICOM_T1_FLAIR_TI_MIN_MS"],
                t1_flair_ti_max=config["DICOM_T1_FLAIR_TI_MAX_MS"],
            ),
            muscle_options={
                "command": config["MUSCLEMAP_COMMAND"], "segmentation_command": config["MUSCLEMAP_SEGMENT_COMMAND"],
                "segmentation_region": config["MUSCLEMAP_SEGMENT_REGION"], "method": config["MUSCLEMAP_METHOD"],
                "region": config["MUSCLEMAP_REGION"], "components": config["MUSCLEMAP_COMPONENTS"],
                "timeout_seconds": config["EXTRACTION_TIMEOUT_SECONDS"], "python_executable": config["MUSCLEMAP_PYTHON"],
                "output_root": config["MUSCLE_SEGMENTATION_DIR"], "workers": config["MUSCLEMAP_GMM_WORKERS"],
            },
            radiomics_options={
                "config_path": config["FAE_RADIOMICS_CONFIG"], "python_executable": config["RADIOMICS_PYTHON"],
                "timeout_seconds": config["EXTRACTION_TIMEOUT_SECONDS"],
                "segmentation_options": {
                    "output_root": config["RADIOMIC_SEGMENTATION_DIR"], "python_executable": config["TOTALSEGMENTATOR_PYTHON"],
                    "weights_path": config["TOTALSEGMENTATOR_WEIGHTS_PATH"], "device": config["TOTALSEGMENTATOR_DEVICE"],
                    "require_gpu": config["TOTALSEGMENTATOR_REQUIRE_GPU"],
                    "nr_threads_resampling": config["TOTALSEGMENTATOR_NR_THREADS_RESAMPLING"],
                    "nr_threads_saving": config["TOTALSEGMENTATOR_NR_THREADS_SAVING"],
                    "timeout_seconds": config["SEGMENTATION_TIMEOUT_SECONDS"],
                },
            },
        )
        response = Response(
            stream_with_context(_stream_dicom_processing(events, temporary)),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
        response.call_on_close(temporary.cleanup)
        return response

    @app.get("/api/schema/vertebra-levels")
    def vertebra_levels():
        """The levels the Radiomics VOI can be built from."""
        return jsonify({
            "levels": available_levels(),
            "default": app.config["RADIOMIC_DEFAULT_LEVELS"],
        })

    @app.post("/api/upload/radiomics")
    def upload_radiomics():
        # Without an uploaded VOI, TotalSegmentator intersects its total_mr
        # vertebral-body mask with the selected vertebrae_mr levels.
        supplied_mask = request.files.get("mask")
        has_mask = supplied_mask is not None and bool(supplied_mask.filename)
        levels = []
        if not has_mask:
            try:
                levels = normalise_levels(request.form.getlist("levels"))
            except ExtractionInputError as exc:
                return jsonify({
                    "error": "invalid_levels",
                    "message": str(exc),
                }), 422
        segmentation_options = {
            "output_root": app.config["RADIOMIC_SEGMENTATION_DIR"],
            "python_executable": app.config["TOTALSEGMENTATOR_PYTHON"],
            "weights_path": app.config["TOTALSEGMENTATOR_WEIGHTS_PATH"],
            "device": app.config["TOTALSEGMENTATOR_DEVICE"],
            "require_gpu": app.config["TOTALSEGMENTATOR_REQUIRE_GPU"],
            "nr_threads_resampling": app.config[
                "TOTALSEGMENTATOR_NR_THREADS_RESAMPLING"
            ],
            "nr_threads_saving": app.config[
                "TOTALSEGMENTATOR_NR_THREADS_SAVING"
            ],
            "timeout_seconds": app.config["SEGMENTATION_TIMEOUT_SECONDS"],
        }
        return _handle_extraction(
            modality="Radiomics",
            extractor=lambda images, mask, expected: (
                extract_radiomics_features_events(
                    images[0],
                    mask,
                    expected,
                    config_path=app.config["FAE_RADIOMICS_CONFIG"],
                    python_executable=app.config["RADIOMICS_PYTHON"],
                    timeout_seconds=app.config["EXTRACTION_TIMEOUT_SECONDS"],
                    levels=levels,
                    segmentation_options=segmentation_options,
                )
            ),
            predictor=predictor,
            mask_mode="optional",
        )

    @app.post("/api/predict")
    def predict():
        if predictor is None:
            return _model_unavailable(model_load_error)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({
                "error": "invalid_json",
                "message": "Request body must be a JSON object.",
            }), 400

        clinical = payload.get("clinical", {})
        if clinical is None:
            clinical = {}
        if not isinstance(clinical, dict):
            return jsonify({
                "error": "invalid_clinical_payload",
                "message": "clinical must be a JSON object.",
            }), 422
        if not clinical:
            return jsonify({
                "error": "clinical_required",
                "message": (
                    "Complete all Clinical fields before prediction. Muscle "
                    "and Radiomics inputs may be uploaded in any order."
                ),
                "details": {"required_modality": "Clinical"},
            }), 422

        # Extracted features travel with the request; the server keeps no
        # per-upload state between extraction and prediction.
        modality_features = {}
        payload_errors = {}
        for modality in ("Muscle", "Radiomics"):
            supplied = payload.get(modality.lower())
            if supplied in (None, "", {}):
                continue
            if not isinstance(supplied, dict):
                payload_errors[modality.lower()] = (
                    f"{modality} must be a JSON object of feature values."
                )
                continue
            modality_features[modality] = supplied

        if payload_errors:
            return jsonify({
                "error": "invalid_features",
                "message": "One or more supplied feature sets are invalid.",
                "details": payload_errors,
            }), 422

        try:
            result = predictor.predict(
                clinical=clinical,
                muscle=modality_features.get("Muscle"),
                radiomics=modality_features.get("Radiomics"),
            )
        except PredictionInputError as exc:
            return jsonify({
                "error": "incomplete_features",
                "message": str(exc),
                "details": exc.details,
            }), 422
        except BundleValidationError as exc:
            app.logger.exception("Stored model failed during prediction")
            return jsonify({
                "error": "model_inference_error",
                "message": str(exc),
            }), 500
        except Exception:
            app.logger.exception("Unexpected prediction failure")
            return jsonify({
                "error": "prediction_failed",
                "message": "The model could not complete this prediction.",
            }), 500
        return jsonify(result)

    @app.post("/api/explain")
    def explain():
        """Stream exactly one explanation of the selected late-fusion prediction."""
        if predictor is None or explainer is None:
            return _model_unavailable(model_load_error)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({
                "error": "invalid_json",
                "message": "Request body must be a JSON object.",
            }), 400
        try:
            prepared = explainer.prepare(payload)
        except PredictionInputError as exc:
            return jsonify({
                "error": "incomplete_features", "message": str(exc), "details": exc.details,
            }), 422
        except ExplanationUnavailableError as exc:
            return jsonify({"error": "explanation_unavailable", "message": str(exc)}), 422
        except Exception:
            app.logger.exception("Failed to prepare late-fusion SHAP")
            return jsonify({
                "error": "explanation_failed",
                "message": "The late-fusion prediction could not be prepared for SHAP.",
            }), 500

        def generate():
            yield _ndjson({"event": "start", "modality": "Late Fusion"})
            try:
                yield _ndjson({"event": "explanation", **explainer.explain(prepared)})
                yield _ndjson({"event": "complete", "modality": "Late Fusion"})
            except ExplanationUnavailableError as exc:
                yield _ndjson({
                    "event": "error", "modality": "Late Fusion",
                    "error": "explanation_unavailable", "message": str(exc),
                })
            except ExplanationError as exc:
                yield _ndjson({
                    "event": "error", "modality": "Late Fusion",
                    "error": "explanation_failed", "message": str(exc),
                })
            except Exception:
                app.logger.exception("Late-fusion SHAP failed")
                yield _ndjson({
                    "event": "error", "modality": "Late Fusion",
                    "error": "explanation_failed",
                    "message": "The final late-fusion prediction could not be explained.",
                })

        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )
    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(exc):
        maximum_mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return jsonify({
            "error": "upload_too_large",
            "message": f"Upload exceeds the {maximum_mb:g} MB limit.",
        }), 413

    return app


def _handle_extraction(
        modality: str,
        extractor: Callable,
        predictor: LateFusionPredictor | None,
        mask_mode: str = "required",
):
    """Stream one extraction.

    ``mask_mode`` is ``"required"``, ``"optional"`` (Radiomics, which segments
    the image itself when no VOI is uploaded) or ``"never"`` (Muscle, which
    always generates its own VOI).
    """
    if predictor is None:
        return _model_unavailable(
            "A loaded model bundle is required to determine selected features."
        )
    # Muscle accepts every axial T2 series of one case; Radiomics stays a
    # single image, with or without its VOI mask.
    image_uploads = [
        upload for upload in request.files.getlist("image") if upload and upload.filename
    ]
    mask_upload = request.files.get("mask")
    if mask_upload is not None and not mask_upload.filename:
        mask_upload = None
    missing_fields = []
    if not image_uploads:
        missing_fields.append("image")
    if mask_mode == "required" and mask_upload is None:
        missing_fields.append("mask")
    if missing_fields:
        return jsonify({
            "error": "missing_files",
            "message": {
                "required": "Both NIfTI image and VOI mask are required.",
                "optional": "A Sagittal T1w MRI NIfTI image is required.",
                "never": "At least one axial T2w MRI NIfTI series is required.",
            }[mask_mode],
            "missing_fields": missing_fields,
        }), 400
    if mask_mode == "never":
        # Muscle generates its own VOI for every series.
        mask_upload = None
    else:
        image_uploads = image_uploads[:1]

    safe_image_names = [secure_filename(upload.filename) for upload in image_uploads]
    safe_mask_name = (
        secure_filename(mask_upload.filename) if mask_upload is not None else None
    )
    invalid_files = [
        name
        for name in (*safe_image_names, safe_mask_name)
        if name is not None and (not name or not _allowed_upload(name))
    ]
    if invalid_files:
        return jsonify({
            "error": "unsupported_file_type",
            "message": "Each supplied image/mask must be a .nii or .nii.gz NIfTI file.",
            "invalid_files": invalid_files,
        }), 415

    expected_features = predictor.expected_features(modality)
    # Werkzeug closes the multipart temp files once the view returns, so the
    # uploads must be buffered before the streamed body starts producing.
    events = extractor(
        [_buffer_upload(upload) for upload in image_uploads],
        _buffer_upload(mask_upload),
        expected_features,
    )
    return Response(
        stream_with_context(
            _stream_extraction(
                modality,
                events,
                expected_features,
                safe_image_names,
                safe_mask_name,
            )
        ),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _extraction_result(
        modality: str,
        features: dict,
        expected_features,
        image_filenames,
        mask_filename: str | None,
        segmentation: dict | None = None,
):
    missing_features = [
        feature for feature in expected_features if feature not in features
    ]
    ready = not missing_features
    if not ready:
        message = f"{modality} extraction is missing selected model features."
    elif segmentation:
        levels = ", ".join(segmentation.get("levels", [])) or "the selected levels"
        message = (
            f"TotalSegmentator built a body-only VOI for {levels} and "
            "PyRadiomics extracted "
            f"{len(features)} selected {modality} features from that VOI."
        )
    else:
        message = (
            f"Extracted {len(features)} selected {modality} features from "
            f"{len(image_filenames)} series."
        )
    return {
        "modality": modality,
        "image_filenames": list(image_filenames),
        "image_filename": image_filenames[0] if image_filenames else None,
        "series_count": len(image_filenames),
        "mask_filename": mask_filename,
        "segmentation": segmentation,
        "status": "ready" if ready else "incomplete",
        "features_ready": ready,
        "features": features,
        "missing_features": missing_features,
        "message": message,
    }


def _buffer_upload(upload: FileStorage | None) -> FileStorage | None:
    if upload is None:
        return None
    return FileStorage(
        stream=io.BytesIO(upload.read()),
        filename=upload.filename,
        content_type=upload.content_type,
    )


def _stream_extraction(
        modality: str,
        events,
        expected_features,
        image_filenames,
        mask_filename: str | None,
):
    """Emit one JSON object per line: stage events, then a result or an error."""
    # The extractor drops MuscleMap's own outputs from the selection, so the
    # series it announces are the ones the result should name.
    processed = []
    segmentation = None
    try:
        while True:
            try:
                stage = next(events)
            except StopIteration as stop:
                yield _ndjson({
                    "event": "result",
                    **_extraction_result(
                        modality, stop.value, expected_features,
                        processed or image_filenames, mask_filename,
                        segmentation,
                    ),
                })
                return
            if stage.get("stage") == "series_start" and stage.get("filename"):
                processed.append(secure_filename(stage["filename"]))
            if stage.get("stage") == "segmented":
                segmentation = _segmentation_with_file_urls(stage.get("segmentation"))
                stage = {**stage, "segmentation": segmentation}
            yield _ndjson({"event": "stage", **stage})
    except ExtractorUnavailableError as exc:
        yield _ndjson({
            "event": "error", "error": "extractor_unavailable", "message": str(exc),
        })
    except (ExtractionInputError, ExtractionExecutionError) as exc:
        yield _ndjson({
            "event": "error",
            "error": "feature_extraction_failed",
            "message": str(exc),
        })
    except Exception:
        current_app.logger.exception("Unexpected %s extraction failure", modality)
        yield _ndjson({
            "event": "error",
            "error": "feature_extraction_failed",
            "message": f"Unexpected {modality} extraction failure.",
        })


def _model_unavailable(model_load_error: str | None):
    return jsonify({
        "error": "model_unavailable",
        "message": model_load_error or "Model bundle is not loaded.",
    }), 503


def _stream_dicom_processing(events, temporary):
    try:
        for event in events:
            if event.get("segmentation"):
                event = {**event, "segmentation": _segmentation_with_file_urls(event["segmentation"])}
            yield _ndjson(event)
    except (ExtractionInputError, ExtractionExecutionError, ExtractorUnavailableError) as exc:
        yield _ndjson({"event": "error", "error": "dicom_processing_failed", "message": str(exc)})
    except Exception:
        current_app.logger.exception("Unexpected DICOM processing failure")
        yield _ndjson({"event": "error", "error": "dicom_processing_failed", "message": "The DICOM case could not be processed."})
    finally:
        events.close()
        temporary.cleanup()


def _segmentation_root(source):
    config_key = {"musclemap": "MUSCLE_SEGMENTATION_DIR", "totalsegmentator": "RADIOMIC_SEGMENTATION_DIR"}.get(source)
    return Path(current_app.config[config_key]).expanduser().resolve() if config_key else None


def _segmentation_with_file_urls(segmentation):
    """Give this extraction links to its saved artifacts, scoped by a signed token."""
    if not segmentation:
        return segmentation
    root = _segmentation_root(segmentation.get("source"))
    directory = Path(segmentation.get("directory") or "").resolve()
    urls = {}
    if root is not None and directory != root and directory.is_relative_to(root):
        signer = current_app.extensions["extraction_file_signer"]
        for key, filename in segmentation.get("files", {}).items():
            path = (directory / filename).resolve()
            if path.is_relative_to(directory) and path.is_file():
                token = signer.dumps({"source": segmentation["source"], "directory": str(directory), "filename": filename})
                urls[key] = url_for("extraction_file", token=token)
    return {**segmentation, "file_urls": urls}


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5050")),
        debug=False,
    )
