from PyQt6 import QtWidgets, QtCore
from typing import Optional, Type

from pycroglia.core.io.geometry_export import GeometryExportSelection
from pycroglia.core.io.output import OutputWriter
from pycroglia.ui.widgets.common.labeled_widgets import LabeledLineEdit
from pycroglia.ui.widgets.io.folder_selector import FolderSelector
from pycroglia.ui.widgets.results.writer import OutputWriterSelector


class OutputConfigurator(QtWidgets.QWidget):
    """Widget for tabular writers, destination folder, filename, and geometry export.

    Attributes:
        writer_selector (OutputWriterSelector): Tabular output format checkboxes.
        folder_selector (FolderSelector): Destination folder.
        filename_input (LabeledLineEdit): Base name for tabular files (required if a writer is selected).
    """

    DEFAULT_SAVE_BUTTON_TXT = "Save"
    DEFAULT_FILENAME_PLACEHOLDER_TXT = "Filename"

    # Folder path, filename, list of writers
    buttonCliched = QtCore.pyqtSignal(str, str, object)

    def __init__(
        self,
        writer_widget_title: str,
        folder_selection_label: str,
        folder_button_txt: str,
        folder_path_display_text: str,
        folder_dialog_title: str,
        folder_dialog_path: str = QtCore.QDir.homePath(),
        save_button_txt: Optional[str] = None,
        filename_placeholder: Optional[str] = None,
        writers: Optional[Type[OutputWriter]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Initialize the OutputConfigurator widget.

        Args:
            writer_widget_title (str): Title text shown above the writer selection widget.
            folder_selection_label (str): Label text describing the folder selection section.
            folder_button_txt (str): Text displayed on the button that opens the folder dialog.
            folder_path_display_text (str): Placeholder text for the folder path display field.
            folder_dialog_title (str): Title used for the folder selection dialog window.
            folder_dialog_path (str): Initial directory path when the folder dialog opens.
            writers (Optional[Type[OutputWriter]]): Collection or type providing available writer classes.
                If None, uses OutputWriter.get_writers().
            parent (Optional[QtWidgets.QWidget]): Optional parent widget.
        """
        super().__init__(parent=parent)

        # State
        self.results_ready = False

        # Widgets
        self.writer_selector = OutputWriterSelector(
            writers=writers if writers else OutputWriter.get_writers(),
            title_text=writer_widget_title,
            parent=self,
        )
        self.folder_selector = FolderSelector(
            label_text=folder_selection_label,
            button_text=folder_button_txt,
            path_display_text=folder_path_display_text,
            dialog_title=folder_dialog_title,
            dialog_path=folder_dialog_path,
            parent=self,
        )

        self.filename_input = LabeledLineEdit(
            label_text=filename_placeholder or self.DEFAULT_FILENAME_PLACEHOLDER_TXT,
            parent=self,
        )
        self.button = QtWidgets.QPushButton(
            save_button_txt or self.DEFAULT_SAVE_BUTTON_TXT, parent=self
        )
        self.button.setEnabled(False)

        self.button.clicked.connect(self._on_button_clicked)
        self.folder_selector.folderSelected.connect(self._on_status_changed)
        self.writer_selector.itemChanged.connect(self._on_status_changed)
        self.filename_input.valueChanged.connect(self._on_status_changed)

        self._geometry_skeleton = self._make_format_row(
            "Skeleton (surface mesh)",
            ("sk_obj", "sk_ply", "sk_vtp", "sk_vtk"),
        )
        self._geometry_mask = QtWidgets.QGroupBox("Cell mask (VTK only)", parent=self)
        mask_layout = QtWidgets.QVBoxLayout()
        self._geometry_mask_surface = QtWidgets.QCheckBox(
            "Surface mesh (marching cubes → PolyData .vtk)", parent=self._geometry_mask
        )
        self._geometry_mask_vol = QtWidgets.QCheckBox(
            "Binary mask volume (ImageData .vtk, no polygon mesh)", parent=self._geometry_mask
        )
        self._geometry_mask_surface.stateChanged.connect(self._on_status_changed)
        self._geometry_mask_vol.stateChanged.connect(self._on_status_changed)
        mask_layout.addWidget(self._geometry_mask_surface)
        mask_layout.addWidget(self._geometry_mask_vol)
        self._geometry_mask.setLayout(mask_layout)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.writer_selector)
        layout.addWidget(self.folder_selector)
        layout.addWidget(self.filename_input)
        layout.addWidget(self._geometry_skeleton["group"])
        layout.addWidget(self._geometry_mask)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def set_results_ready(self, ready: bool):
        self.results_ready = ready
        self._on_status_changed()

    def _make_format_row(
        self, title: str, attr_names: tuple[str, str, str, str]
    ) -> dict:
        group = QtWidgets.QGroupBox(title, parent=self)
        row = QtWidgets.QHBoxLayout()
        labels = ("OBJ", "PLY", "VTP", "VTK")
        checks = []
        for label, aname in zip(labels, attr_names):
            cb = QtWidgets.QCheckBox(label, parent=group)
            cb.stateChanged.connect(self._on_status_changed)
            row.addWidget(cb)
            checks.append((aname, cb))
        group.setLayout(row)
        return {"group": group, "checks": checks}

    def _geometry_any_checked(self) -> bool:
        for _, cb in self._geometry_skeleton["checks"]:
            if cb.isChecked():
                return True
        if self._geometry_mask_surface.isChecked():
            return True
        return self._geometry_mask_vol.isChecked()

    def get_geometry_export_selection(self) -> GeometryExportSelection:
        def _get(aname: str) -> bool:
            for name, cb in self._geometry_skeleton["checks"]:
                if name == aname:
                    return cb.isChecked()
            return False

        return GeometryExportSelection(
            skeleton_obj=_get("sk_obj"),
            skeleton_ply=_get("sk_ply"),
            skeleton_vtp=_get("sk_vtp"),
            skeleton_vtk=_get("sk_vtk"),
            mask_vtk=self._geometry_mask_surface.isChecked(),
            mask_vtk_volume=self._geometry_mask_vol.isChecked(),
        )

    def _on_status_changed(self):
        writers_on = self.writer_selector.has_selected_writers()
        filename_ok = self.filename_input.has_text()
        tabular_ok = (not writers_on) or filename_ok
        output_ok = writers_on or self._geometry_any_checked()
        if (
            self.folder_selector.has_folder_selected()
            and tabular_ok
            and output_ok
            and self.results_ready
        ):
            self.button.setEnabled(True)
        else:
            self.button.setEnabled(False)

    def _on_button_clicked(self):
        folder = self.folder_selector.get_selected_folder()
        file_name = self.filename_input.get_text()
        writers = self.writer_selector.get_selected_writers()

        self.buttonCliched.emit(folder, file_name, writers)
