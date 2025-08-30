from typing import Optional, Any
from PyQt6 import QtWidgets

from pycroglia.ui.widgets.wizard.config import DEFAULT_CONFIG, create_wizard_pages
from pycroglia.ui.widgets.wizard.pages import PageManager


class ConfigurableMainStack(QtWidgets.QWidget):
    """Main widget that manages the application workflow using page abstractions.

    Accepts text configuration through a dictionary for internationalization and customization.
    Orchestrates the complete workflow from file selection through filtering to segmentation.

    Attributes:
        stacked (QtWidgets.QStackedWidget): Stacked widget for page navigation.
        page_manager (PageManager): Manager for handling page navigation and data flow.
        config (dict[str, Any]): Configuration dictionary for text and UI elements.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = DEFAULT_CONFIG.copy(),
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the configurable main wizard stack.

        Args:
            config (Optional[dict[str, Any]]): Configuration dictionary for text and UI elements.
                If None, uses DEFAULT_CONFIG. Expected structure matches DEFAULT_CONFIG
                with sections for 'file_selection', 'filter_editor', 'segmentation_editor',
                and 'navigation'.
            parent (Optional[QtWidgets.QWidget]): Parent widget.
        """
        super().__init__(parent)

        # Merge user config with defaults
        self.config = config

        # Widgets
        self.stacked = QtWidgets.QStackedWidget(self)
        self.page_manager = PageManager(self.stacked, self)

        # Create all pages using configuration
        self._create_pages()

        # Layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.stacked)
        self.setLayout(main_layout)

    def _create_pages(self):
        """Create and add all wizard pages using configuration.

        Creates pages in the following order:
        1. File selection page
        2. Filter editor page
        3. Segmentation editor page

        Each page is configured with the appropriate text elements from self.config.
        """
        page_configs = create_wizard_pages(self.config)

        for page_config in page_configs:
            widget = page_config["widget_class"](
                parent=self, **page_config["widget_args"]
            )
            page = page_config["page_class"](widget)

            # Add page with navigation
            self.page_manager.add_page(page, **page_config["navigation"])
