#!/usr/bin/env python3
"""
Complete XER Data Extractor - Extracts ALL data from Primavera P6 XER files
No information is skipped - this is a comprehensive deep extraction system
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional


class CompleteXERExtractor:
    """
    Comprehensive XER file parser that extracts EVERY piece of information
    including all tables, relationships, custom fields, and metadata
    """

    def __init__(self, filepath: str, file_type: str = "baseline"):
        """
        Initialize the complete extractor

        Args:
            filepath: Path to XER file
            file_type: "baseline" or "update"
        """
        self.filepath = Path(filepath)
        self.file_type = file_type
        self.filename = self.filepath.name

        # Core data structures - store EVERYTHING
        self.raw_content = ""
        self.metadata = {}
        self.tables = {}  # All tables with all fields
        self.table_relationships = defaultdict(list)  # Track relationships between tables
        self.extraction_stats = {}
        self.parsing_errors = []

    def extract_all(self) -> 'CompleteXERExtractor':
        """
        Main extraction method - extracts EVERYTHING from the XER file
        """
        print(f"Starting complete extraction of {self.filename}...")

        # Read raw file content
        self._read_raw_content()

        # Parse header metadata
        self._parse_header()

        # Parse all tables with all fields
        self._parse_all_tables()

        # Build relationship maps
        self._build_relationships()

        # Calculate extraction statistics
        self._calculate_statistics()

        print(f"Extraction complete: {len(self.tables)} tables, {sum(len(t) for t in self.tables.values())} total records")

        return self

    def _read_raw_content(self):
        """Read the complete raw file content"""
        try:
            with open(self.filepath, 'r', encoding='windows-1252', errors='ignore') as f:
                self.raw_content = f.read()
        except Exception as e:
            self.parsing_errors.append(f"Error reading file: {str(e)}")
            raise

    def _parse_header(self):
        """Parse XER file header with all metadata"""
        lines = self.raw_content.split('\n')
        if not lines:
            return

        header_line = lines[0].strip()
        parts = header_line.split('\t')

        if parts[0] == 'ERMHDR':
            self.metadata = {
                'format': 'ERMHDR',
                'version': parts[1] if len(parts) > 1 else '',
                'export_date': parts[2] if len(parts) > 2 else '',
                'export_user_type': parts[3] if len(parts) > 3 else '',
                'export_username': parts[4] if len(parts) > 4 else '',
                'export_user_fullname': parts[5] if len(parts) > 5 else '',
                'database_name': parts[6] if len(parts) > 6 else '',
                'user_role': parts[7] if len(parts) > 7 else '',
                'currency_symbol': parts[8] if len(parts) > 8 else '',
                'file_type': self.file_type,
                'filename': self.filename,
                'file_size_bytes': len(self.raw_content),
                'file_size_mb': round(len(self.raw_content) / (1024 * 1024), 2)
            }

    def _parse_all_tables(self):
        """Parse ALL tables with ALL fields - nothing is skipped"""
        lines = self.raw_content.split('\n')

        current_table = None
        current_fields = []

        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')

            try:
                if parts[0] == '%T':  # Table declaration
                    current_table = parts[1] if len(parts) > 1 else ''
                    self.tables[current_table] = []
                    current_fields = []

                elif parts[0] == '%F':  # Field names
                    current_fields = parts[1:]

                elif parts[0] == '%R':  # Row data - capture ALL fields
                    if current_table and current_fields:
                        row_data = {
                            '_table_name': current_table,
                            '_row_number': len(self.tables[current_table]) + 1
                        }

                        # Capture ALL fields, even empty ones
                        for i, field in enumerate(current_fields):
                            value = parts[i+1] if i+1 < len(parts) else ''
                            row_data[field] = value

                        self.tables[current_table].append(row_data)

            except Exception as e:
                self.parsing_errors.append(f"Line {line_num}: {str(e)}")

    def _build_relationships(self):
        """Build comprehensive relationship maps between all tables"""

        # Task relationships
        if 'TASKPRED' in self.tables:
            for pred in self.tables['TASKPRED']:
                self.table_relationships['TASK_PREDECESSORS'].append({
                    'task_id': pred.get('task_id', ''),
                    'predecessor_task_id': pred.get('pred_task_id', ''),
                    'relationship_type': pred.get('pred_type', ''),
                    'lag_hours': pred.get('lag_hr_cnt', '0'),
                    'lag_days': float(pred.get('lag_hr_cnt', 0)) / 8 if pred.get('lag_hr_cnt') else 0
                })

        # Resource assignments
        if 'TASKRSRC' in self.tables:
            for rsrc in self.tables['TASKRSRC']:
                self.table_relationships['TASK_RESOURCES'].append({
                    'task_id': rsrc.get('task_id', ''),
                    'resource_id': rsrc.get('rsrc_id', ''),
                    'role_id': rsrc.get('role_id', ''),
                    'budgeted_cost': rsrc.get('target_cost', ''),
                    'actual_cost': rsrc.get('act_reg_cost', ''),
                    'remaining_cost': rsrc.get('remain_cost', '')
                })

        # WBS hierarchy
        if 'PROJWBS' in self.tables:
            for wbs in self.tables['PROJWBS']:
                self.table_relationships['WBS_HIERARCHY'].append({
                    'wbs_id': wbs.get('wbs_id', ''),
                    'parent_wbs_id': wbs.get('parent_wbs_id', ''),
                    'wbs_name': wbs.get('wbs_name', ''),
                    'wbs_short_name': wbs.get('wbs_short_name', '')
                })

        # Activity codes
        if 'TASKACTV' in self.tables:
            for actv in self.tables['TASKACTV']:
                self.table_relationships['TASK_ACTIVITY_CODES'].append({
                    'task_id': actv.get('task_id', ''),
                    'activity_code_id': actv.get('actv_code_id', ''),
                    'activity_code_type_id': actv.get('actv_code_type_id', '')
                })

    def _calculate_statistics(self):
        """Calculate comprehensive statistics about the extraction"""
        self.extraction_stats = {
            'total_tables': len(self.tables),
            'total_records': sum(len(t) for t in self.tables.values()),
            'file_type': self.file_type,
            'file_size_mb': self.metadata.get('file_size_mb', 0),
            'parsing_errors': len(self.parsing_errors),
            'extraction_timestamp': datetime.now().isoformat()
        }

        # Table-level statistics
        self.extraction_stats['tables'] = {}
        for table_name, records in self.tables.items():
            if records:
                field_count = len(records[0]) - 2  # Exclude metadata fields
                self.extraction_stats['tables'][table_name] = {
                    'record_count': len(records),
                    'field_count': field_count
                }

    def get_complete_data(self) -> Dict[str, Any]:
        """
        Get ALL extracted data in a structured format
        Returns EVERYTHING - no data is excluded
        """
        return {
            'metadata': self.metadata,
            'tables': self.tables,
            'relationships': dict(self.table_relationships),
            'statistics': self.extraction_stats,
            'parsing_errors': self.parsing_errors,

            # Convenience accessors for common data
            'project': self.get_project_info(),
            'tasks': self.get_all_tasks(),
            'resources': self.get_all_resources(),
            'calendars': self.get_all_calendars(),
            'wbs': self.get_wbs_structure(),
            'activity_codes': self.get_activity_codes(),
            'custom_fields': self.get_custom_fields(),
            'relationships_summary': self.get_relationships_summary()
        }

    def get_project_info(self) -> Dict[str, Any]:
        """Extract comprehensive project information"""
        if 'PROJECT' not in self.tables or not self.tables['PROJECT']:
            return {}

        proj = self.tables['PROJECT'][0]
        return {
            'project_id': proj.get('proj_id', ''),
            'project_name': proj.get('proj_short_name', ''),
            'full_name': proj.get('proj_short_name', ''),
            'plan_start_date': proj.get('plan_start_date', ''),
            'plan_end_date': proj.get('plan_end_date', ''),
            'scheduled_end_date': proj.get('scd_end_date', ''),
            'data_date': proj.get('last_recalc_date', ''),
            'actual_start_date': proj.get('act_start_date', ''),
            'actual_end_date': proj.get('act_end_date', ''),
            'status_code': proj.get('status_code', ''),
            'critical_path_type': proj.get('critical_path_type', ''),
            'total_float_hours': proj.get('total_float_hr_cnt', ''),
            'orig_cost': proj.get('orig_cost', ''),
            'indep_remain_cost': proj.get('indep_remain_cost', ''),
            'project_flag': proj.get('project_flag', ''),
            'all_fields': proj  # Include ALL fields
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get ALL tasks with ALL fields"""
        if 'TASK' not in self.tables:
            return []

        return self.tables['TASK']  # Return everything, let downstream decide what to use

    def get_all_resources(self) -> List[Dict[str, Any]]:
        """Get ALL resources with ALL fields"""
        if 'RSRC' not in self.tables:
            return []

        return self.tables['RSRC']

    def get_all_calendars(self) -> List[Dict[str, Any]]:
        """Get ALL calendars with ALL fields"""
        if 'CALENDAR' not in self.tables:
            return []

        return self.tables['CALENDAR']

    def get_wbs_structure(self) -> List[Dict[str, Any]]:
        """Get complete WBS structure"""
        if 'PROJWBS' not in self.tables:
            return []

        return self.tables['PROJWBS']

    def get_activity_codes(self) -> Dict[str, Any]:
        """Get all activity codes and their assignments"""
        activity_codes = {
            'types': self.tables.get('ACTVTYPE', []),
            'values': self.tables.get('ACTVCODE', []),
            'assignments': self.tables.get('TASKACTV', [])
        }
        return activity_codes

    def get_custom_fields(self) -> Dict[str, Any]:
        """Get all custom field definitions and values"""
        return {
            'definitions': self.tables.get('UDFTYPE', []),
            'values': self.tables.get('UDFVALUE', [])
        }

    def get_relationships_summary(self) -> Dict[str, int]:
        """Get summary of all relationships"""
        return {
            'predecessor_links': len(self.table_relationships.get('TASK_PREDECESSORS', [])),
            'resource_assignments': len(self.table_relationships.get('TASK_RESOURCES', [])),
            'wbs_items': len(self.table_relationships.get('WBS_HIERARCHY', [])),
            'activity_code_assignments': len(self.table_relationships.get('TASK_ACTIVITY_CODES', []))
        }

    def save_to_json(self, output_path: str):
        """Save ALL extracted data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_complete_data(), f, indent=2, ensure_ascii=False)
        print(f"Complete data saved to: {output_path}")

    def generate_extraction_report(self) -> str:
        """Generate a human-readable report of what was extracted"""
        report_lines = [
            "="*80,
            f"COMPLETE XER EXTRACTION REPORT",
            f"File: {self.filename}",
            f"Type: {self.file_type}",
            "="*80,
            "",
            "EXTRACTION STATISTICS:",
            f"  Total Tables Extracted: {self.extraction_stats['total_tables']}",
            f"  Total Records Extracted: {self.extraction_stats['total_records']}",
            f"  File Size: {self.metadata.get('file_size_mb', 0)} MB",
            f"  Parsing Errors: {len(self.parsing_errors)}",
            "",
            "TABLES EXTRACTED (with record counts):"
        ]

        for table_name in sorted(self.tables.keys()):
            count = len(self.tables[table_name])
            report_lines.append(f"  {table_name:30s} : {count:6d} records")

        report_lines.extend([
            "",
            "RELATIONSHIPS MAPPED:",
            f"  Predecessor Links: {len(self.table_relationships.get('TASK_PREDECESSORS', []))}",
            f"  Resource Assignments: {len(self.table_relationships.get('TASK_RESOURCES', []))}",
            f"  WBS Hierarchy Items: {len(self.table_relationships.get('WBS_HIERARCHY', []))}",
            f"  Activity Code Assignments: {len(self.table_relationships.get('TASK_ACTIVITY_CODES', []))}",
            "",
            "="*80
        ])

        return "\n".join(report_lines)


def extract_complete_xer_data(filepath: str, file_type: str = "baseline") -> CompleteXERExtractor:
    """
    Convenience function to extract all data from an XER file

    Args:
        filepath: Path to XER file
        file_type: "baseline" or "update"

    Returns:
        CompleteXERExtractor with all data extracted
    """
    extractor = CompleteXERExtractor(filepath, file_type)
    extractor.extract_all()
    return extractor


if __name__ == "__main__":
    # Test extraction on all XER files in current directory
    xer_files = list(Path('.').glob('*.xer'))

    if not xer_files:
        print("No XER files found in current directory!")
    else:
        for xer_file in xer_files:
            print(f"\n{'='*80}")
            print(f"Processing: {xer_file.name}")
            print('='*80)

            extractor = extract_complete_xer_data(str(xer_file), "baseline")
            print(extractor.generate_extraction_report())

            # Save to JSON
            output_file = xer_file.stem + "_complete_data.json"
            extractor.save_to_json(output_file)
