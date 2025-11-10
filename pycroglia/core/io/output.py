import json
import re
import unicodedata
import inspect
import numbers
import numpy as np

from abc import ABC
from dataclasses import dataclass
from openpyxl import Workbook
from typing import Any, Optional, List


@dataclass
class AnalysisSummary:
    """Summary statistics for microglia analysis.

    Attributes:
        file: Name of the analyzed file.
        avg_centroid_distance: Average distance between cell centroids.
        total_territorial_volume: Total volume of all cell territories.
        total_unoccupied_volume: Volume not occupied by any cell territory.
        percent_occupied_volume: Percentage of total volume that is occupied.
    """

    file: str
    avg_centroid_distance: float
    total_territorial_volume: float
    total_unoccupied_volume: float
    percent_occupied_volume: float


@dataclass
class AnalysisSummaryConfig:
    """Configuration for column headers in analysis summary output.

    Attributes:
        file_txt: Header text for file column.
        avg_centroid_distance_txt: Header text for average centroid distance.
        total_territorial_volume_txt: Header text for total territorial volume.
        total_unoccupied_volume_txt: Header text for total unoccupied volume.
        percent_occupied_volume_txt: Header text for percent occupied volume.
    """

    file_txt: str
    avg_centroid_distance_txt: str
    total_territorial_volume_txt: str
    total_unoccupied_volume_txt: str
    percent_occupied_volume_txt: str

    @classmethod
    def default(cls):
        """Create default configuration with standard column headers.

        Returns:
            AnalysisSummaryConfig: Configuration with default header texts.
        """
        return cls(
            file_txt="File",
            avg_centroid_distance_txt="Avg Centroid Distance",
            total_territorial_volume_txt="TotMgTerritoryVol",
            total_unoccupied_volume_txt="TotUnoccupiedVol",
            percent_occupied_volume_txt="PercentOccupiedVol",
        )


@dataclass
class CellAnalysis:
    """Analysis results for an individual microglia cell.

    Attributes:
        cell_territory_volume: Volume of the cell's territory.
        cell_volume: Volume of the cell itself.
        ramification_index: Measure of cell branching complexity.
        number_of_endpoints: Count of branch endpoints.
        number_of_branches: Count of branch points.
        avg_branch_length: Average length of all branches.
        max_branch_length: Length of the longest branch.
        min_branch_length: Length of the shortest branch.
    """

    cell_territory_volume: float
    cell_volume: float
    ramification_index: float
    number_of_endpoints: int
    number_of_branches: int
    avg_branch_length: float
    max_branch_length: float
    min_branch_length: float
    branch_analysis: dict[str, Any]
    full_cell_analysis: dict[str, Any]


@dataclass()
class CellAnalysisConfig:
    """Configuration for column headers in per-cell analysis output.

    Attributes:
        cell_territory_volume_txt: Header text for cell territory volume.
        cell_volume_txt: Header text for cell volume.
        ramification_index_txt: Header text for ramification index.
        number_of_endpoints_txt: Header text for number of endpoints.
        number_of_branches_txt: Header text for number of branches.
        avg_branch_length_txt: Header text for average branch length.
        max_branch_length_txt: Header text for maximum branch length.
        min_branch_length_txt: Header text for minimum branch length.
    """

    cell_territory_volume_txt: str
    cell_volume_txt: str
    ramification_index_txt: str
    number_of_endpoints_txt: str
    number_of_branches_txt: str
    avg_branch_length_txt: str
    max_branch_length_txt: str
    min_branch_length_txt: str

    @classmethod
    def default(cls):
        """Create default configuration with standard column headers.

        Returns:
            CellAnalysisConfig: Configuration with default header texts.
        """
        return cls(
            cell_territory_volume_txt="CellTerritoryVol",
            cell_volume_txt="CellVolumes",
            ramification_index_txt="RamificationIndex",
            number_of_endpoints_txt="NumOfEndpoints",
            number_of_branches_txt="NumOfBranchpoints",
            avg_branch_length_txt="AvgBranchLength",
            max_branch_length_txt="MaxBranchLength",
            min_branch_length_txt="MinBranchLength",
        )


@dataclass
class FullAnalysis:
    """Complete analysis results containing summary and per-cell data.

    Attributes:
        summary: Overall analysis summary statistics.
        cells: List of individual cell analysis results.
    """

    summary: AnalysisSummary
    cells: List[CellAnalysis]


class OutputWriter(ABC):
    """Abstract base class for writing analysis results to files."""

    def write(self, file_path: str, data: FullAnalysis):
        """Write analysis data to a file.

        Args:
            file_path: Path where the output file should be saved.
            data: Complete analysis results to write.
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """Return the display name of the writer.

        Returns:
            str: Name of the writer.
        """
        raise NotImplementedError

    @classmethod
    def get_writers(cls):
        """Return all non-abstract subclasses of OutputWriter.

        Returns:
            list: List of OutputWriter subclasses.
        """
        return [c for c in cls.__subclasses__() if not inspect.isabstract(c)]


class ExcelOutput(OutputWriter):
    """Excel output writer for microglia analysis results.

    Creates Excel workbooks with separate sheets for summary statistics
    and per-cell analysis data.
    """

    DEFAULT_SUMMARY_SHEET_TITLE = "Summary"
    DEFAULT_PER_CELL_SHEET_TITLE = "PerCell"
    DEFAULT_FILE_EXTENSION = ".xlsx"
    DEFAULT_NAME = "Multi Sheet Excel"

    def __init__(
        self,
        summary_title: Optional[str] = None,
        per_cell_title: Optional[str] = None,
        summary_config: Optional[AnalysisSummaryConfig] = None,
        per_cell_config: Optional[CellAnalysisConfig] = None,
    ):
        """Initialize Excel output writer with custom configurations.

        Args:
            summary_title: Custom title for the summary sheet.
            per_cell_title: Custom title for the per-cell data sheet.
            summary_config: Configuration for summary column headers.
            per_cell_config: Configuration for per-cell column headers.
        """
        super().__init__()

        self.summary_title = summary_title or self.DEFAULT_SUMMARY_SHEET_TITLE
        self.per_cell_title = per_cell_title or self.DEFAULT_PER_CELL_SHEET_TITLE

        self.summary_config = summary_config or AnalysisSummaryConfig.default()
        self.per_cell_config = per_cell_config or CellAnalysisConfig.default()

    @classmethod
    def get_name(cls) -> str:
        """Return the display name of the Excel writer.

        Returns:
            str: Name of the writer.
        """
        return cls.DEFAULT_NAME

    def write(self, file_path: str, data: FullAnalysis):
        """Write analysis data to an Excel file.

        Args:
            file_path: Path where the Excel file should be saved.
            data: Complete analysis results to write.
        """
        complete_path = self._create_path(file_path)
        wb = Workbook()

        self._write_summary_sheet(wb, data.summary)
        self._write_per_cell_sheet(wb, data.cells)

        wb.save(complete_path)

    def _create_path(self, path: str):
        """Ensure the file path has the correct Excel extension.

        Args:
            path: Original file path.

        Returns:
            str: File path with .xlsx extension if not already present.
        """
        if not path.lower().endswith(".xlsx"):
            path += self.DEFAULT_FILE_EXTENSION
        return path

    def _write_summary_sheet(self, wb: Workbook, summary: AnalysisSummary):
        """Write summary statistics to the summary sheet.

        Args:
            wb: Excel workbook to write to.
            summary: Summary statistics to write.
        """
        ws_summary = wb.active
        ws_summary.title = self.summary_title
        ws_summary.append([self.summary_config.file_txt, summary.file])
        ws_summary.append(
            [
                self.summary_config.avg_centroid_distance_txt,
                float(summary.avg_centroid_distance),
            ]
        )
        ws_summary.append(
            [
                self.summary_config.total_territorial_volume_txt,
                float(summary.total_territorial_volume),
            ]
        )
        ws_summary.append(
            [
                self.summary_config.total_unoccupied_volume_txt,
                float(summary.total_unoccupied_volume),
            ]
        )
        ws_summary.append(
            [
                self.summary_config.percent_occupied_volume_txt,
                float(summary.percent_occupied_volume),
            ]
        )

    def _write_per_cell_sheet(self, wb: Workbook, per_cell: list[CellAnalysis]):
        """Write per-cell analysis data to the per-cell sheet.

        Args:
            wb: Excel workbook to write to.
            per_cell: List of individual cell analysis results.
        """
        ws_per_cell = wb.create_sheet(title=self.per_cell_title)
        headers = [
            "Index",
            self.per_cell_config.cell_territory_volume_txt,
            self.per_cell_config.cell_volume_txt,
            self.per_cell_config.ramification_index_txt,
            self.per_cell_config.number_of_endpoints_txt,
            self.per_cell_config.number_of_branches_txt,
            self.per_cell_config.avg_branch_length_txt,
            self.per_cell_config.max_branch_length_txt,
            self.per_cell_config.min_branch_length_txt,
        ]

        ws_per_cell.append(headers)
        for i, cell in enumerate(per_cell):
            ws_per_cell.append(
                [
                    int(i),
                    float(cell.cell_territory_volume),
                    float(cell.cell_volume),
                    float(cell.ramification_index),
                    int(cell.number_of_endpoints),
                    int(cell.number_of_branches),
                    float(cell.avg_branch_length),
                    float(cell.max_branch_length),
                    float(cell.min_branch_length),
                ]
            )


class JSONOutput(OutputWriter):
    """JSON output writer for microglia analysis results.

    Serializes analysis data to JSON format using configured field names
    for consistency with other output formats.
    """

    DEFAULT_FILE_EXTENSION = ".json"
    DEFAULT_NAME = "JSON File"

    def __init__(
        self,
        summary_config: Optional[AnalysisSummaryConfig] = None,
        per_cell_config: Optional[CellAnalysisConfig] = None,
        indent: Optional[int] = 4,
    ):
        """Initialize JSON output writer with custom configurations.

        Args:
            summary_config: Configuration for summary field names.
            per_cell_config: Configuration for per-cell field names.
            indent: Number of spaces for JSON indentation. None for compact output.
        """
        super().__init__()
        self.summary_config = summary_config or AnalysisSummaryConfig.default()
        self.per_cell_config = per_cell_config or CellAnalysisConfig.default()
        self.indent = indent

    @classmethod
    def get_name(cls) -> str:
        """Return the display name of the JSON writer.

        Returns:
            str: Name of the writer.
        """
        return cls.DEFAULT_NAME

    def write(self, file_path: str, data: FullAnalysis):
        """Write analysis data to a JSON file.

        Args:
            file_path: Path where the JSON file should be saved.
            data: Complete analysis results to write.
        """
        complete_path = self._create_path(file_path)
        json_data = self._convert_to_dict(data)

        # Normalize any non-native JSON types (e.g. numpy.uint64, numpy arrays)
        json_data = self._normalize_json_compatible(json_data)

        with open(complete_path, "w") as f:
            json.dump(json_data, f, indent=self.indent)

    def _normalize_json_compatible(self, obj: Any) -> Any:
        """Recursively convert objects into JSON-serializable native Python types.

        This handles numpy scalar types (e.g. uint64, int64, float32), numpy arrays,
        and traverses lists/tuples/sets/dicts to normalize nested structures.

        Args:
            obj: The object to normalize.

        Returns:
            A JSON-serializable representation of obj using native Python types.
        """
        # Primitives that json handles directly
        if obj is None or isinstance(obj, (str, bool)):
            return obj

        if isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
            return float(obj)

        # Numpy-specific conversions when numpy is available
        if np is not None:
            # numpy scalar (np.integer, np.floating, np.bool_)
            if isinstance(obj, np.generic):
                # np.bool_ is subclass of np.generic but not numbers.Integral/Real reliably
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)

            # numpy arrays -> lists
            if isinstance(obj, np.ndarray):
                return self._normalize_json_compatible(obj.tolist())

        # Collections
        if isinstance(obj, dict):
            return {str(k): self._normalize_json_compatible(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [self._normalize_json_compatible(v) for v in obj]

        # Fallback for dataclass-like or other objects: try to convert via dict or attributes
        # If it's dataclass-like (has __dict__), normalize that
        try:
            if hasattr(obj, "__dict__"):
                return self._normalize_json_compatible(vars(obj))
        except Exception:
            pass

        # Last resort: convert to string
        return str(obj)

    def _create_path(self, path: str) -> str:
        """Ensure the file path has the correct JSON extension.

        Args:
            path: Original file path.

        Returns:
            str: File path with .json extension if not already present.
        """
        if not path.lower().endswith(".json"):
            path += self.DEFAULT_FILE_EXTENSION
        return path

    def _convert_to_dict(self, data: FullAnalysis) -> dict:
        """Convert FullAnalysis to dictionary using configured field names.

        Args:
            data: Complete analysis results to convert.

        Returns:
            dict: Dictionary representation with configured field names.
        """
        return {
            "summary": self._convert_summary_to_dict(data.summary),
            "cells": [self._convert_cell_to_dict(cell) for cell in data.cells],
        }

    def _to_snake_case(self, text: str) -> str:
        """Convert text to lowercase snake_case format.

        Args:
            text: Text to convert.

        Returns:
            str: Text converted to lowercase snake_case.
        """
        # Normalize unicode characters to remove accents
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "_", text)
        text = text.lower()
        text = re.sub(r"_+", "_", text)
        text = text.strip("_")
        return text

    def _convert_summary_to_dict(self, summary: AnalysisSummary) -> dict:
        """Convert summary using configuration field names.

        Args:
            summary: Summary statistics to convert.

        Returns:
            dict: Summary with configured field names in snake_case.
        """
        return {
            self._to_snake_case(self.summary_config.file_txt): summary.file,
            self._to_snake_case(
                self.summary_config.avg_centroid_distance_txt
            ): summary.avg_centroid_distance,
            self._to_snake_case(
                self.summary_config.total_territorial_volume_txt
            ): summary.total_territorial_volume,
            self._to_snake_case(
                self.summary_config.total_unoccupied_volume_txt
            ): summary.total_unoccupied_volume,
            self._to_snake_case(
                self.summary_config.percent_occupied_volume_txt
            ): summary.percent_occupied_volume,
        }

    def _convert_cell_to_dict(self, cell: CellAnalysis) -> dict:
        """Convert cell analysis using configuration field names.

        Args:
            cell: Individual cell analysis to convert.

        Returns:
            dict: Cell analysis with configured field names in snake_case.
        """
        return {
            self._to_snake_case(
                self.per_cell_config.cell_territory_volume_txt
            ): cell.cell_territory_volume,
            self._to_snake_case(self.per_cell_config.cell_volume_txt): cell.cell_volume,
            self._to_snake_case(
                self.per_cell_config.ramification_index_txt
            ): cell.ramification_index,
            self._to_snake_case(
                self.per_cell_config.number_of_endpoints_txt
            ): cell.number_of_endpoints,
            self._to_snake_case(
                self.per_cell_config.number_of_branches_txt
            ): cell.number_of_branches,
            self._to_snake_case(
                self.per_cell_config.avg_branch_length_txt
            ): cell.avg_branch_length,
            self._to_snake_case(
                self.per_cell_config.max_branch_length_txt
            ): cell.max_branch_length,
            self._to_snake_case(
                self.per_cell_config.min_branch_length_txt
            ): cell.min_branch_length,
        }
