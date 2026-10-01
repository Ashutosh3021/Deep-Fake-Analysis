"""
Gunicorn config -- picked up automatically because the start command runs
`cd backend && gunicorn ... api:app`.

Why this file exists:

  With --preload the app is imported in the gunicorn *master* and then forked.
  Any daemon thread started at import time (our model warm-up) dies with the
  fork, so every worker came up cold. The first upload then loaded every model
  inline, and on a small instance that took longer than the proxy allowed --
  the browser saw a 502.

  post_fork runs inside the freshly forked worker, before it accepts traffic,
  so warm-up restarts there.
"""


def post_fork(server, worker):  # noqa: ARG001 (gunicorn hook signature)
    try:
        import api
    except Exception as exc:  # pragma: no cover - import errors must not kill boot
        server.log.warning("post_fork: could not import api (%s)", exc)
        return
    try:
        api.ensure_warmup()
        server.log.info("post_fork: model warm-up started in worker %s", worker.pid)
    except Exception as exc:  # pragma: no cover
        server.log.warning("post_fork: warm-up failed (%s)", exc)
