from abc import ABC
from dataclasses import dataclass
from typing import Optional, List
from openpyxl import Workbook


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
        raise NotImplemented


class ExcelOutput(OutputWriter):
    """Excel output writer for microglia analysis results.

    Creates Excel workbooks with separate sheets for summary statistics
    and per-cell analysis data.
    """

    DEFAULT_SUMMARY_SHEET_TITLE = "Summary"
    DEFAULT_PER_CELL_SHEET_TITLE = "PerCell"
    DEFAULT_FILE_EXTENSION = ".xlsx"

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
        for cell in per_cell:
            ws_per_cell.append(
                [
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
