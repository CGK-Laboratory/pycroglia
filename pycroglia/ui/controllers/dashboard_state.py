import copy
import numpy as np

from typing import Optional, List, Any
from numpy.typing import NDArray
from PyQt6 import QtCore, QtWidgets

import pycroglia.core.centroid as centroids
import pycroglia.core.territorial_volume as t_volume
import pycroglia.core.full_cell_analysis as f_cell
import pycroglia.core.branch_analysis as b_analysis


from pycroglia.core.compute.qt_pool import QPool
from pycroglia.core.io.output import (
    AnalysisSummaryConfig,
    CellAnalysisConfig,
    CellAnalysis,
    AnalysisSummary,
)


class ResultsDashboardTextConfig(QtCore.QObject):
    @staticmethod
    def _make_default_summary_config() -> AnalysisSummaryConfig:
        return AnalysisSummaryConfig(
            file_txt="",
            avg_centroid_distance_txt="Average centroid distance (μm)",
            total_territorial_volume_txt="Total territorial volume (μm³)",
            total_unoccupied_volume_txt="Total unoccupied volume (μm³)",
            percent_occupied_volume_txt="Percent occupied volume (μm³)",
        )

    @staticmethod
    def _make_default_per_cell_config() -> CellAnalysisConfig:
        return CellAnalysisConfig(
            cell_territory_volume_txt="Cell territory volume (μm³)",
            cell_volume_txt="Cell volume (μm³)",
            ramification_index_txt="Ramification index",
            number_of_endpoints_txt="Number of endpoints",
            number_of_branches_txt="Number of branches",
            avg_branch_length_txt="Avg branch length (μm)",
            max_branch_length_txt="Max branch length (μm)",
            min_branch_length_txt="Min branch length (μm)",
        )

    def __init__(
        self,
        file: str,
        summary_config: Optional[AnalysisSummaryConfig] = None,
        per_cell_config: Optional[CellAnalysisConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        # Configuration
        self._file = file
        self._summary = summary_config or self._make_default_summary_config()
        self._per_cell = per_cell_config or self._make_default_per_cell_config()
        self._summary.file_txt = self._file

    def get_summary_text_config(self) -> AnalysisSummaryConfig:
        return self._summary

    def get_per_cell_text_config(self) -> CellAnalysisConfig:
        return self._per_cell


class QCounter(QtCore.QObject):
    reached = QtCore.pyqtSignal()

    def __init__(self, n: int, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)

        self._lock = QtCore.QMutex()
        self._total_number = n
        self._count = 0

    @QtCore.pyqtSlot()
    def _emit_reached(self):
        self.reached.emit()

    def add(self):
        locker = QtCore.QMutexLocker(self._lock)
        try:
            self._count += 1
            if self._count >= self._total_number:
                QtCore.QMetaObject.invokeMethod(
                    self, "_emit_reached", QtCore.Qt.ConnectionType.QueuedConnection
                )
        finally:
            del locker


class MetricsDAG(QtCore.QObject):
    resultsAvailable = QtCore.pyqtSignal()
    progressUpdated = QtCore.pyqtSignal(int, int)  # (completed, total)

    def __init__(
        self,
        masks: List[NDArray],
        scale: float = 1.0,
        z_scale: float = 1.0,
        vox_scale: float = 1.0,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        # State
        self._finished = False

        # Pool
        self._qt_pool = QPool()
        self._qt_branch_pool = QPool()

        # Progress counters
        self._total_tasks = 0
        self._completed_tasks = 0

        # Barriers
        self._all_tasks = QCounter(4)
        self._all_cells = QCounter(len(masks))

        # Results
        self._centroids = {}
        self._territorial_volume = {}
        self._full_cell_analysis = {}
        self._branch_analysis = {}

        # Parameters
        self._masks = copy.copy(masks)
        self._scale = scale
        self._z_scale = z_scale
        self._vox_scale = vox_scale
        self._n_cells = len(self._masks)
        if self._n_cells == 0:
            raise ValueError("No image to process")
        self._z_planes = self._masks[0].shape[0]

        # Connections
        self._all_tasks.reached.connect(self._all_tasks_finished)
        self._all_cells.reached.connect(self._all_branches_finished)

    def _on_qpool_task_finished(self, task_id: str) -> None:
        """Internal slot invoked when any QPool task finishes.

        Updates internal completed counter and emits aggregated progress.
        """
        self._completed_tasks += 1
        self.progressUpdated.emit(self._completed_tasks, self._total_tasks)

    @QtCore.pyqtSlot(dict)
    def _add_centroids_results(self, result: dict[str, Any]):
        self._centroids = result

        self._all_tasks.add()

        for i in range(self._n_cells):
            mask = self._masks[i]

            analysis = b_analysis.BranchAnalysis(
                cell=copy.copy(mask),
                centroid=self._centroids[centroids.KEY_CENTROIDS][i],
                scale=self._scale,
                zscale=self._z_scale,
                zslices=self._z_planes,
            )

            # count branch tasks and provide a finished callback so we can track progress
            self._total_tasks += 1
            self._qt_branch_pool.submit(
                computable=analysis,
                on_result=lambda res, idx=i: self._add_branch_result(idx, res),
                on_error=lambda msg, exc, idx=i: self._on_branch_analysis_error(
                    msg, exc, idx
                ),
                on_finish=self._on_qpool_task_finished,
            )

        self._qt_branch_pool.run()

    def cancel(self):
        self._qt_pool.cancel()
        self._qt_branch_pool.cancel()

    @QtCore.pyqtSlot(dict)
    def _add_territorial_volume_results(self, result: dict[str, Any]):
        self._territorial_volume = result
        self._all_tasks.add()

    @QtCore.pyqtSlot(dict)
    def _add_full_analysis_results(self, result: dict[str, Any]):
        self._full_cell_analysis = result
        self._all_tasks.add()

    @QtCore.pyqtSlot(int, dict)
    def _add_branch_result(self, idx: int, result: dict[str, Any]):
        self._branch_analysis[idx] = result
        self._all_cells.add()

    @QtCore.pyqtSlot(str, Exception, int)
    def _on_branch_analysis_error(self, msg: str, exc: Exception, idx: int):
        self._branch_analysis[idx] = b_analysis.get_empty_branch_analysis()
        self._all_cells.add()

    @QtCore.pyqtSlot()
    def _all_branches_finished(self):
        self._all_tasks.add()

    @QtCore.pyqtSlot()
    def _all_tasks_finished(self):
        self._finished = True
        self.resultsAvailable.emit()

    def run(self):
        # reset counters for a fresh run
        self._total_tasks = 0
        self._completed_tasks = 0

        list_of_computables = [
            (
                centroids.Centroids(
                    masks=self._masks, scale=self._scale, zscale=self._z_scale
                ),
                self._add_centroids_results,
            ),
            (
                t_volume.TerritorialVolume(
                    masks=self._masks, voxscale=self._vox_scale, zplanes=self._z_planes
                ),
                self._add_territorial_volume_results,
            ),
            (
                f_cell.FullCellAnalysis(masks=self._masks, voxscale=self._vox_scale),
                self._add_full_analysis_results,
            ),
        ]

        # submit main computables and count them
        for computable in list_of_computables:
            self._total_tasks += 1
            self._qt_pool.submit(
                computable=computable[0],
                on_result=computable[1],
                on_finish=self._on_qpool_task_finished,
            )

        self._qt_pool.run()

    def get_analysis_summary(self, file: str) -> AnalysisSummary:
        avg_centroid_distance = (
            self._centroids.get(centroids.KEY_AVG_CENTROIDS_DISTANCE, 0.0)
            if isinstance(self._centroids, dict)
            else 0.0
        )

        territorial_volume = self._territorial_volume or {}
        total_territorial_volume = float(
            territorial_volume.get(t_volume.KEY_TOTAL_VOLUME_COVERED, 0.0)
        )
        total_unoccupied_volume = float(
            territorial_volume.get(t_volume.KEY_EMPTY_VOLUME, 0.0)
        )
        percent_occupied_volume = float(
            territorial_volume.get(t_volume.KEY_COVERED_PERCENTAGE, 0.0)
        )

        return AnalysisSummary(
            file=file,
            avg_centroid_distance=avg_centroid_distance,
            total_territorial_volume=total_territorial_volume,
            total_unoccupied_volume=total_unoccupied_volume,
            percent_occupied_volume=percent_occupied_volume,
        )

    def get_per_cell_analysis(self) -> List[CellAnalysis]:
        per_cell = []

        cells_convex = list(
            (self._territorial_volume or {}).get(
                t_volume.KEY_CONVEX_VOLUME, [0.0] * self._n_cells
            )
        )
        branch_results = self._branch_analysis or {}
        full_cell_analysis = (
            self._full_cell_analysis
            if self._full_cell_analysis and isinstance(self._full_cell_analysis, dict)
            else {}
        )

        cell_volumes = list(
            full_cell_analysis.get(f_cell.KEY_CELL_VOLUMES, [0.0] * self._n_cells)
        )
        cell_complexities = list(
            full_cell_analysis.get(f_cell.KEY_CELL_COMPLEXITIES, [0.0] * self._n_cells)
        )

        for i in range(self._n_cells):
            cell_territory_volume = (
                float(cells_convex[i]) if i < len(cells_convex) else 0.0
            )
            cell_volume = float(cell_volumes[i]) if i < len(cell_volumes) else 0.0

            ramification_index = (
                float(cell_complexities[i]) if i < len(cell_complexities) else 0.0
            )

            number_of_branches = branch_results[i].get(
                b_analysis.KEY_NUM_BRANCHPOINTS, 0
            )
            number_of_endpoints = np.sum(
                branch_results[i].get(b_analysis.KEY_ENDPOINTS, 0)
            )

            avg_branch_length = branch_results[i].get(
                b_analysis.KEY_AVG_BRANCH_LENGTH, 0.0
            )
            max_branch_length = branch_results[i].get(
                b_analysis.KEY_MAX_BRANCH_LENGTH, 0.0
            )
            min_branch_length = branch_results[i].get(
                b_analysis.KEY_MIN_BRANCH_LENGTH, 0.0
            )

            per_cell.append(
                CellAnalysis(
                    cell_territory_volume=cell_territory_volume,
                    cell_volume=cell_volume,
                    ramification_index=ramification_index,
                    number_of_branches=number_of_branches,
                    number_of_endpoints=number_of_endpoints,
                    avg_branch_length=avg_branch_length,
                    max_branch_length=max_branch_length,
                    min_branch_length=min_branch_length,
                    full_cell_analysis=full_cell_analysis,
                    branch_analysis=branch_results,
                )
            )

        return per_cell


class ResultsDashboardState(QtCore.QObject):
    resultsChanged = QtCore.pyqtSignal()
    progressChanged = QtCore.pyqtSignal(int, int)  # forwarded (completed, total)

    resultsChanged = QtCore.pyqtSignal()

    @staticmethod
    def _make_default_summary() -> AnalysisSummary:
        return AnalysisSummary(
            avg_centroid_distance=0,
            total_territorial_volume=0,
            total_unoccupied_volume=0,
            percent_occupied_volume=0,
            file="",
        )

    @staticmethod
    def _make_default_per_cell() -> List[CellAnalysis]:
        return [
            CellAnalysis(
                cell_territory_volume=0,
                cell_volume=0,
                ramification_index=0,
                number_of_endpoints=0,
                number_of_branches=0,
                avg_branch_length=0,
                max_branch_length=0,
                min_branch_length=0,
                full_cell_analysis={},
                branch_analysis={},
            )
        ]

    def __init__(
        self,
        file: str,
        img: NDArray,
        cells_masks: List[NDArray],
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)

        # Images
        self._file = file
        self._img = img
        self._cells_masks = cells_masks

        # Initial state
        self._summary_state = self._make_default_summary()
        self._summary_state.file = self._file
        self._per_cell_state = self._make_default_per_cell()

        # Executions
        self._execution: Optional[MetricsDAG] = None

    def calculate_results(
        self, scale: float = 1.0, z_scale: float = 1.0, vox_scale: float = 1.0
    ):
        self._execution = MetricsDAG(
            masks=self._cells_masks,
            scale=scale,
            z_scale=z_scale,
            vox_scale=vox_scale,
            parent=self.parent(),
        )

        # forward progress events to outside consumers
        self._execution.progressUpdated.connect(
            lambda c, t: self.progressChanged.emit(c, t)
        )

        self._execution.resultsAvailable.connect(self._handle_available_results)
        self._execution.run()

    @QtCore.pyqtSlot()
    def _handle_available_results(self):
        self._summary_state = self._execution.get_analysis_summary(self._file)
        self._per_cell_state = self._execution.get_per_cell_analysis()

        self._execution.cancel()
        self._execution = None

        self.resultsChanged.emit()

    def get_summary(self) -> AnalysisSummary:
        return self._summary_state

    def get_per_cell(self) -> List[CellAnalysis]:
        return self._per_cell_state
