import os
from logger import logger as log
from webapp.controller import app
from waitress import serve
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound

def get_application_mode():
    """Get the mode in which the application will be started from environment variables."""
    mode = os.environ.get('MODE', 'dev').lower()
    if mode not in ['prod', 'dev']:
        message = f"Invalid application mode: {mode}. Use 'dev' for development or 'prod' for production."
        log.error(message)
        raise ValueError(message)
    return mode

def get_application_port(default=5000):
    """Get the application port number from environment variables."""
    try:
        return int(os.environ.get('PORT', default))
    except ValueError as e:
        log.error(f"Invalid port number. Falling back to default {default}. Error: {e}")
        return default

def main():
    """Main function to start the model on the server."""
    mode = get_application_mode()
    port = get_application_port()

    log.info(f"Starting application in {mode} mode on port {port}.")

    application = DispatcherMiddleware(NotFound(), {app.config['APPLICATION_ROOT']: app})

    if mode == 'dev':
        run_simple('0.0.0.0', port, application, use_debugger=True, use_reloader=True)
    elif mode == 'prod':
        serve(application, host='0.0.0.0', port=port)

    log.info("Application has stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"Application failed to start due to an error: {e}")
