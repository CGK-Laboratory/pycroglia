from typing import Any, List
from PyQt6 import QtCore

from pycroglia.ui.widgets.imagefilters.stacks import FilterEditorStack
from pycroglia.ui.widgets.io.file_selection_editor import FileSelectionEditor
from pycroglia.ui.widgets.segmentation.stacks import SegmentationEditorStack
from pycroglia.ui.widgets.wizard.pages import (
    FileSelectionPage,
    FilterEditorPage,
    SegmentationEditorPage,
)

DEFAULT_CONFIG = {
    # File selection page
    "file_selection": {
        "file_headers": ["File Type", "File Path"],
        "delete_button_text": "Delete",
        "open_file_text": "Select Files:",
        "open_button_text": "Open",
        "open_dialog_title": "Open File",
        "file_filters": "All Files (*);;Image Files (*.lsm *.tiff *.tif)",
    },
    # Filter editor page
    "filter_editor": {
        "img_viewer_label": "Image Viewer",
        "read_button_text": "Load Image",
        "channels_label": "Channels:",
        "channel_of_interest_label": "Channel of Interest:",
        "gray_filter_label": "Gray Filter",
        "gray_filter_slider_label": "Threshold:",
        "small_objects_filter_label": "Small Objects Filter",
        "small_objects_threshold_label": "Min Size:",
    },
    # Segmentation editor page
    "segmentation_editor": {
        "segmentation_headers": ["Cell Number", "Cell Size"],
        "rollback_button_text": "Roll back segmentation",
        "segmentation_button_text": "Segment Cell",
        "progress_title": "Segmenting cell...",
        "progress_cancel_text": "Cancel",
    },
    # Navigation buttons
    "navigation": {
        "back_button_text": "Back",
        "next_button_text": "Next",
    },
}


def create_wizard_pages(config: dict[str, Any]) -> List[dict[str, Any]]:
    """Create wizard page configurations based on provided config.

    Args:
        config (dict[str, Any]): Configuration dictionary containing text and UI settings
            for all wizard pages. Expected structure:
            - 'file_selection': File selection page configuration
            - 'filter_editor': Filter editor page configuration
            - 'segmentation_editor': Segmentation editor page configuration
            - 'navigation': Navigation button configuration

    Returns:
        List[dict[str, Any]]: List of page configuration dictionaries, each containing:
            - type: Page type identifier
            - widget_class: Widget class to instantiate
            - widget_args: Arguments for widget constructor
            - page_class: Page wrapper class
            - navigation: Navigation button configuration
    """
    page_configs = [
        {
            "type": "file_selection",
            "widget_class": FileSelectionEditor,
            "widget_args": {
                "headers": config["file_selection"]["file_headers"],
                "delete_button_text": config["file_selection"]["delete_button_text"],
                "open_file_text": config["file_selection"]["open_file_text"],
                "open_button_text": config["file_selection"]["open_button_text"],
                "open_dialog_title": config["file_selection"]["open_dialog_title"],
                "open_dialog_default_path": QtCore.QDir.homePath(),
                "file_filters": config["file_selection"]["file_filters"],
            },
            "page_class": FileSelectionPage,
            "navigation": {
                "show_back_btn": False,
                "show_next_btn": True,
                "next_btn_txt": config["navigation"]["next_button_text"],
            },
        },
        {
            "type": "filter_editor",
            "widget_class": FilterEditorStack,
            "widget_args": {
                "img_viewer_label": config["filter_editor"]["img_viewer_label"],
                "read_button_text": config["filter_editor"]["read_button_text"],
                "channels_label": config["filter_editor"]["channels_label"],
                "channel_of_interest_label": config["filter_editor"][
                    "channel_of_interest_label"
                ],
                "gray_filter_label": config["filter_editor"]["gray_filter_label"],
                "gray_filter_slider_label": config["filter_editor"][
                    "gray_filter_slider_label"
                ],
                "small_objects_filter_label": config["filter_editor"][
                    "small_objects_filter_label"
                ],
                "small_objects_threshold_label": config["filter_editor"][
                    "small_objects_threshold_label"
                ],
            },
            "page_class": FilterEditorPage,
            "navigation": {
                "show_back_btn": True,
                "show_next_btn": True,
                "back_btn_txt": config["navigation"]["back_button_text"],
                "next_btn_txt": config["navigation"]["next_button_text"],
            },
        },
        {
            "type": "segmentation_editor",
            "widget_class": SegmentationEditorStack,
            "widget_args": {
                "headers_text": config["segmentation_editor"]["segmentation_headers"],
                "rollback_button_text": config["segmentation_editor"][
                    "rollback_button_text"
                ],
                "segmentation_button_text": config["segmentation_editor"][
                    "segmentation_button_text"
                ],
                "progress_title": config["segmentation_editor"]["progress_title"],
                "progress_cancel_text": config["segmentation_editor"][
                    "progress_cancel_text"
                ],
            },
            "page_class": SegmentationEditorPage,
            "navigation": {
                "show_back_btn": True,
                "show_next_btn": False,
                "back_btn_txt": config["navigation"]["back_button_text"],
            },
        },
    ]

    return page_configs
