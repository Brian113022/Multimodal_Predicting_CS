# Deployment

This folder contains the native deployment CLI, process supervisor and optional
Windows Task Scheduler / macOS LaunchAgent integration. The Flask application
remains in `webapp/`. Use the same Python environment as training and the web app.

## Commands

From the repository root on Windows:

```powershell
deployment\run.bat
deployment\run.bat status
deployment\run.bat stop
deployment\run.bat restart
```

On macOS use `sh deployment/run.sh` with the same arguments. You can also use
`python deployment/deploy.py <action>` or `python -m deployment <action>`.
The default action, `run`, checks local prerequisites, performs Cloudflare tunnel
setup if necessary, stops the managed deployment, then starts a fresh supervisor
and verifies the public endpoint. Running the command again restarts deployment.
`stop` terminates the managed app, tunnel and supervisor and records an explicit
stop so startup tasks cannot revive them until `run` or `start` is used.
`start` keeps an already-running deployment; `restart` stops and starts it.
If an app outside deployment owns the port, `run` reports the conflict instead
of killing that app.
Deployment commands run without interactive yes/no prompts. Setup applies the
configured DNS/tunnel changes automatically after validating the resources.
cloudflared must be installed outside this project, available on `PATH` or through
`CLOUDFLARED_PATH`. Deployment never downloads a local copy. On Windows it also
checks standard shared installation folders when the current terminal has an
outdated `PATH`. Install once with `winget install --id Cloudflare.cloudflared --exact` (Windows) or `brew install cloudflared` (macOS).

For local use without a public tunnel, run `python -m webapp.app` directly.
This direct command does not start the deployment supervisor.

## Configuration and state

Configure `deployment/.env` using `.env.example`. Existing local credentials were
moved with this refactor; no values need to be copied into source code.
`PYTHON_EXECUTABLE` can select the Python environment. Otherwise the launchers
prefer the repository `.venv` and then the active Python.

New installations store private tokens, process state
and logs in `deployment/.deploy/`. These files and `.env` are git-ignored.
For compatibility, an existing `webapp/.deploy/` is reused, and `webapp/.env` is
read only if `deployment/.env` is absent. All deployment entry points live here;
there are no deployment launchers in `webapp/`. If an older startup task points
to `webapp/deploy.py`, unregister it with `autostart-uninstall`, then register
the new entry point with `autostart-install`.
Training data, outputs and model artifact paths are unchanged.

## Recovery after a crash

The supervisor restarts the app and tunnel processes that it owns after they
exit, with a delay that increases up to 60 seconds after repeated failures.
An already-running app started outside deployment is reused and is not forcibly
stopped. A live but hung app is not automatically killed by this supervisor.

To recover the supervisor itself and start it after login, optionally run:

```powershell
deployment\run.bat autostart-install
```

This command registers a Windows Task Scheduler task or macOS LaunchAgent
without a confirmation prompt. Windows retries supervisor failures every minute,
up to the configured 999 retries; macOS uses launchd KeepAlive. This is user-login
startup, not a guarantee that the application runs before login or while the
computer is off/asleep.

Docker is optional. With a containerized deployment, Docker restart policies
such as `unless-stopped` can restart a container after its main process exits.
A health check alone does not restart a running, unhealthy container. See the
[Docker restart policy documentation](https://docs.docker.com/engine/containers/start-containers-automatically/).
This repository's launchers run native Python; this move does not add a Docker image.
