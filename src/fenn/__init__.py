from colorama import Fore, Style
from typing import Callable, Optional, Any, Iterable, Type

from fenn.args import Parser
from fenn.logging import Logger
from fenn.notification import Notifier, Service
from fenn.secrets.keystore import KeyStore
from fenn.utils import generate_haiku_id


class FENN:
    """
    The base FENN application
    """

    def __init__(self) -> None:

        self._session_id: str = generate_haiku_id()

        self._parser: Parser = Parser()
        self._keystore: KeyStore = KeyStore()
        self._logger: Logger = Logger.get_instance()  # SINGLETON
        self._notifier: Notifier = Notifier()
        self._config_file: str = None

        self._entrypoint_fn: Optional[Callable] = None

    def entrypoint(self, entrypoint_fn: Callable) -> Callable:
        """
        The decorator to register the main execution function.
        """
        self._entrypoint_fn = entrypoint_fn
        return entrypoint_fn

    def run(self) -> Any:
        """
        The method that executes the application's core logic.
        """
        self._logger.system_log(
            "***********************************************************************************\n"
            f"{Style.BRIGHT}Hi, thank you for using the {Fore.GREEN}PyFenn{Style.RESET_ALL}{Style.BRIGHT} framework.{Style.RESET_ALL}\n"
            f"PyFenn is still in an {Fore.YELLOW}alpha version{Style.RESET_ALL}.\n"
            "If you find a bug or inconsistency, if you want to contribute or request a feature,\nplease open an issue at "
            f"{Fore.CYAN}https://github.com/pyfenn/fenn/issues{Style.RESET_ALL}\n"
            "Thank you for your support!\n"
            "***********************************************************************************\n"
        )

        if not self._entrypoint_fn:
            raise RuntimeError(
                f"{Fore.RED}[FENN][EXCEPTION] No main function registered. "
                f"Please use {Fore.LIGHTYELLOW_EX}@app.entrypoint{Style.RESET_ALL} "
                "to register your main function."
            )

        # Load config
        self._parser.config_file = (
            self._config_file if self._config_file is not None else "fenn.yaml"
        )
        self._args = self._parser.load_configuration()
        self._args["session_id"] = self._session_id

        # Start logging
        self._logger.start()

        # Print parsed config (user logs)
        self._parser.print()

        try:
            # System startup message
            self._logger.system_info(
                f"Application starting from entrypoint: {self._entrypoint_fn.__name__}"
            )

            # Execute user function
            result = self._entrypoint_fn(self._args)
            return result

        finally:
            self._logger.stop()

    def register_notification_services(
        self,
        services: Iterable[Type[Service]],
    ) -> None:
        """
        Register notification service classes.

        Example:
            app.register_notification_services([Discord, Telegram])
        """

        for service_cls in services:
            self._notifier.add_service(service_cls())

    def notify(self, message: str):
        self._notifier.notify(message)

    @property
    def config_file(self) -> str:
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        """
        The method to set the YAML file.
        """
        self._config_file = config_file

