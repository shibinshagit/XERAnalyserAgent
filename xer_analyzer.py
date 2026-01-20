"""
XER Schedule Analyzer - Robust LLM-Powered Analysis
Comprehensive context and reliable code generation
"""

import re
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class XERDataStore:
    """Stores all XER data with pre-computed statistics"""

    def __init__(self):
        self.baseline = None
        self.updates = []
        self.hours_per_day = 10
        self._cached_stats = None

    def load_baseline(self, data: Dict, name: str, data_date: str):
        """Load baseline data"""
        self.baseline = {
            'name': name,
            'data_date': data_date,
            'data': data,
            'df': self._create_dataframes(data)
        }
        self._cached_stats = None  # Clear cache

    def add_update(self, data: Dict, name: str, data_date: str):
        """Add an update file"""
        self.updates.append({
            'name': name,
            'data_date': data_date,
            'data': data,
            'df': self._create_dataframes(data)
        })
        self.updates.sort(key=lambda x: x['data_date'])
        self._cached_stats = None

    def remove_update(self, index: int):
        """Remove an update by index"""
        if 0 <= index < len(self.updates):
            self.updates.pop(index)
            self._cached_stats = None

    def _create_dataframes(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """Convert XER tables to pandas DataFrames"""
        dfs = {}

        if data.get('tasks'):
            dfs['tasks'] = pd.DataFrame(data['tasks'])

        if data.get('wbs'):
            dfs['wbs'] = pd.DataFrame(data['wbs'])

        for table_name, records in data.get('tables', {}).items():
            if records:
                dfs[table_name.lower()] = pd.DataFrame(records)

        return dfs

    def get_latest(self) -> Dict:
        """Get latest data (most recent update or baseline)"""
        if self.updates:
            return self.updates[-1]
        return self.baseline

    def get_baseline(self) -> Dict:
        """Get baseline data"""
        return self.baseline

    def get_update_by_date(self, date_str: str) -> Optional[Dict]:
        """Find update by date (partial match)"""
        for update in self.updates:
            if date_str in update['data_date'] or date_str in update['name']:
                return update
        return None

    def get_update_by_month(self, month: str, year: str = None) -> Optional[Dict]:
        """Find update by month name or number"""
        month_map = {
            'jan': '01', 'january': '01', '01': '01', '1': '01',
            'feb': '02', 'february': '02', '02': '02', '2': '02',
            'mar': '03', 'march': '03', '03': '03', '3': '03',
            'apr': '04', 'april': '04', '04': '04', '4': '04',
            'may': '05', '05': '05', '5': '05',
            'jun': '06', 'june': '06', '06': '06', '6': '06',
            'jul': '07', 'july': '07', '07': '07', '7': '07',
            'aug': '08', 'august': '08', '08': '08', '8': '08',
            'sep': '09', 'september': '09', '09': '09', '9': '09',
            'oct': '10', 'october': '10', '10': '10',
            'nov': '11', 'november': '11', '11': '11',
            'dec': '12', 'december': '12', '12': '12'
        }

        month_num = month_map.get(month.lower(), month)

        for update in self.updates:
            data_date = update['data_date']
            if len(data_date) >= 7:
                file_month = data_date[5:7]
                file_year = data_date[:4]
                if file_month == month_num:
                    if year is None or file_year == year:
                        return update
        return None

    def compute_basic_stats(self) -> Dict:
        """Compute comprehensive statistics that are always available"""
        if self._cached_stats:
            return self._cached_stats

        source = self.get_latest()
        if not source or 'tasks' not in source.get('df', {}):
            return {'error': 'No data loaded'}

        tasks_df = source['df']['tasks'].copy()
        stats = {}

        # Basic counts
        stats['total_activities'] = len(tasks_df)
        stats['data_source'] = source['name']
        stats['data_date'] = source['data_date']

        # Task types
        if 'task_type' in tasks_df.columns:
            type_counts = tasks_df['task_type'].value_counts().to_dict()
            stats['task_types'] = type_counts
            stats['milestones'] = type_counts.get('TT_Mile', 0) + type_counts.get('TT_FinMile', 0)
            stats['loe_activities'] = type_counts.get('TT_LOE', 0)
            stats['regular_tasks'] = type_counts.get('TT_Task', 0)

        # Status breakdown
        if 'status_code' in tasks_df.columns:
            status_counts = tasks_df['status_code'].value_counts().to_dict()
            stats['status_breakdown'] = status_counts
            stats['completed'] = status_counts.get('TK_Complete', 0)
            stats['in_progress'] = status_counts.get('TK_Active', 0)
            stats['not_started'] = status_counts.get('TK_NotStart', 0)

        # Duration analysis
        if 'target_drtn_hr_cnt' in tasks_df.columns:
            tasks_df['duration_hrs'] = pd.to_numeric(tasks_df['target_drtn_hr_cnt'], errors='coerce').fillna(0)
            tasks_df['duration_days'] = tasks_df['duration_hrs'] / self.hours_per_day

            # Exclude LOE for duration stats
            work_tasks = tasks_df[~tasks_df.get('task_type', '').isin(['TT_LOE', 'TT_Mile', 'TT_FinMile'])]
            if len(work_tasks) > 0:
                stats['long_duration_count'] = len(work_tasks[work_tasks['duration_days'] > 30])
                stats['avg_duration_days'] = round(work_tasks['duration_days'].mean(), 1)
                stats['max_duration_days'] = round(work_tasks['duration_days'].max(), 1)

        # Float/Critical path
        if 'total_float_hr_cnt' in tasks_df.columns:
            tasks_df['float_hrs'] = pd.to_numeric(tasks_df['total_float_hr_cnt'], errors='coerce').fillna(0)
            tasks_df['float_days'] = tasks_df['float_hrs'] / self.hours_per_day

            work_tasks = tasks_df[~tasks_df.get('task_type', '').isin(['TT_LOE'])]
            if len(work_tasks) > 0:
                critical = work_tasks[work_tasks['float_hrs'] <= 0]
                near_critical = work_tasks[(work_tasks['float_hrs'] > 0) & (work_tasks['float_hrs'] <= 100)]
                negative_float = work_tasks[work_tasks['float_hrs'] < 0]

                stats['critical_count'] = len(critical)
                stats['critical_pct'] = round(len(critical) / len(work_tasks) * 100, 1)
                stats['near_critical_count'] = len(near_critical)
                stats['negative_float_count'] = len(negative_float)

        # Relationships
        if 'taskpred' in source['df']:
            pred_df = source['df']['taskpred']
            stats['total_relationships'] = len(pred_df)

            if 'pred_type' in pred_df.columns:
                rel_types = pred_df['pred_type'].value_counts().to_dict()
                stats['relationship_types'] = rel_types

            if 'lag_hr_cnt' in pred_df.columns:
                pred_df['lag'] = pd.to_numeric(pred_df['lag_hr_cnt'], errors='coerce').fillna(0)
                stats['relationships_with_lag'] = len(pred_df[pred_df['lag'] > 0])
                stats['negative_lags'] = len(pred_df[pred_df['lag'] < 0])

            # Open-ended and dangling
            all_task_ids = set(tasks_df['task_id'].tolist())
            has_successor = set(pred_df['pred_task_id'].tolist())
            has_predecessor = set(pred_df['task_id'].tolist())

            no_successor = all_task_ids - has_successor
            no_predecessor = all_task_ids - has_predecessor

            # Exclude LOE and milestones
            work_task_ids = set(tasks_df[~tasks_df['task_type'].isin(['TT_LOE', 'TT_Mile', 'TT_FinMile'])]['task_id'].tolist())
            stats['open_ended_count'] = len(no_successor & work_task_ids)
            stats['dangling_count'] = len(no_predecessor & work_task_ids)

        # Constraints
        if 'cstr_type' in tasks_df.columns:
            constrained = tasks_df[tasks_df['cstr_type'].notna() & (tasks_df['cstr_type'] != '')]
            stats['constrained_activities'] = len(constrained)
            if len(constrained) > 0:
                stats['constraint_types'] = constrained['cstr_type'].value_counts().to_dict()

        # Resources
        if 'taskrsrc' in source['df']:
            rsrc_df = source['df']['taskrsrc']
            stats['resource_assignments'] = len(rsrc_df)
            tasks_with_resources = rsrc_df['task_id'].nunique()
            stats['tasks_with_resources'] = tasks_with_resources
            stats['resource_loaded_pct'] = round(tasks_with_resources / len(tasks_df) * 100, 1)

        # Date range
        if 'target_start_date' in tasks_df.columns:
            starts = tasks_df['target_start_date'].dropna()
            if len(starts) > 0:
                stats['project_start'] = str(starts.min())[:10]

        if 'target_end_date' in tasks_df.columns:
            ends = tasks_df['target_end_date'].dropna()
            if len(ends) > 0:
                stats['project_finish'] = str(ends.max())[:10]

        # Calendars
        if 'calendar' in source['df']:
            cal_df = source['df']['calendar']
            stats['calendar_count'] = len(cal_df)
            if 'clndr_name' in cal_df.columns:
                stats['calendars'] = cal_df['clndr_name'].tolist()

        # Files info
        stats['baseline_name'] = self.baseline['name'] if self.baseline else None
        stats['baseline_date'] = self.baseline['data_date'] if self.baseline else None
        stats['update_count'] = len(self.updates)
        stats['updates'] = [{'name': u['name'], 'date': u['data_date']} for u in self.updates]

        self._cached_stats = stats
        return stats


class XERQueryExecutor:
    """Executes Python code safely against XER data"""

    def __init__(self, data_store: XERDataStore):
        self.data_store = data_store

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute generated code and return results"""
        try:
            latest_data = self.data_store.get_latest()
            baseline_data = self.data_store.get_baseline()

            context = {
                'pd': pd,
                'datetime': datetime,
                'json': json,
                'baseline': baseline_data,
                'updates': self.data_store.updates,
                'latest': latest_data,
                'get_latest': self.data_store.get_latest,
                'get_baseline': self.data_store.get_baseline,
                'get_update_by_date': self.data_store.get_update_by_date,
                'get_update_by_month': self.data_store.get_update_by_month,
                'hours_per_day': self.data_store.hours_per_day,
                'result': None
            }

            exec(code, context)
            result = context.get('result')

            if isinstance(result, pd.DataFrame):
                if len(result) > 50:
                    result = result.head(50)
                result = result.to_dict('records')

            return {'success': True, 'result': result}

        except Exception as e:
            return {'success': False, 'error': str(e)}


class XERAnalyzer:
    """Main analyzer with comprehensive LLM support"""

    def __init__(self):
        self.data_store = XERDataStore()
        self.executor = XERQueryExecutor(self.data_store)

    def load_baseline(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get('project', {}).get('project_name', 'Baseline')
        if data_date is None:
            data_date = data.get('project', {}).get('data_date', '')[:10]
        self.data_store.load_baseline(data, name, data_date)

    def add_update(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get('project', {}).get('project_name', 'Update')
        if data_date is None:
            data_date = data.get('project', {}).get('data_date', '')[:10]
        self.data_store.add_update(data, name, data_date)

    def remove_update(self, index: int):
        self.data_store.remove_update(index)

    def get_basic_stats(self) -> Dict:
        """Get pre-computed statistics"""
        return self.data_store.compute_basic_stats()

    def execute_code(self, code: str) -> Dict:
        return self.executor.execute(code)

    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM"""
        return """You are an expert Primavera P6 Schedule Analyst AI Assistant. You analyze construction project schedules from XER files and provide professional, insightful analysis.

YOUR CAPABILITIES:
1. Analyze schedule quality metrics
2. Compare baseline vs update files
3. Identify schedule issues and risks
4. Provide actionable recommendations
5. Answer any question about the schedule data

RESPONSE STYLE:
- Be direct, professional, and precise
- Include specific numbers and percentages
- Highlight key findings and concerns
- Provide recommendations when appropriate
- Use bullet points for clarity
- Format tables for comparisons

ANALYSIS APPROACH:
- Always consider schedule best practices
- Flag potential issues (negative float, open-ended activities, hard constraints)
- Interpret data in context of construction project management
- Suggest improvements when you see problems"""

    def get_code_generation_prompt(self, user_query: str, basic_stats: Dict) -> str:
        """Generate prompt for code generation"""

        source = self.data_store.get_latest()

        # Get column names from actual data
        columns_info = ""
        if source and 'df' in source:
            for table_name, df in list(source['df'].items())[:8]:
                cols = list(df.columns)[:15]
                columns_info += f"\n{table_name.upper()}: {', '.join(cols)}"

        return f"""Generate Python code to answer this question about a Primavera P6 schedule.

USER QUESTION: {user_query}

CURRENT PROJECT STATISTICS (always available as fallback):
{json.dumps(basic_stats, indent=2, default=str)}

AVAILABLE DATA TABLES AND COLUMNS:{columns_info}

AVAILABLE VARIABLES IN CODE:
- latest: dict with latest schedule data, access DataFrames via latest['df']['table_name']
- baseline: dict with baseline data, access via baseline['df']['table_name']
- updates: list of update dicts
- get_update_by_month('feb'): returns update for that month
- get_update_by_month('mar'): returns update for that month
- pd: pandas library
- hours_per_day: 10 (for duration conversion)

KEY FIELDS:
- tasks DataFrame: task_id, task_name, task_code, task_type, status_code, target_start_date, target_end_date, act_start_date, act_end_date, target_drtn_hr_cnt (duration in HOURS), total_float_hr_cnt (float in HOURS), phys_complete_pct, cstr_type, cstr_date, wbs_id, clndr_id
- taskpred DataFrame: task_id, pred_task_id, pred_type (PR_FS, PR_SS, PR_FF, PR_SF), lag_hr_cnt
- taskrsrc DataFrame: task_id, rsrc_id, target_qty, act_reg_qty, target_cost, act_reg_cost
- projwbs DataFrame: wbs_id, wbs_name, parent_wbs_id
- calendar DataFrame: clndr_id, clndr_name, day_hr_cnt, week_hr_cnt

TASK TYPES: TT_Task (regular), TT_Mile (start milestone), TT_FinMile (finish milestone), TT_LOE (level of effort)
STATUS CODES: TK_NotStart, TK_Active, TK_Complete
CONSTRAINT TYPES: CS_MSOA, CS_MEOA, CS_MEOB, CS_MSO, CS_MEO

IMPORTANT RULES:
1. Always use .copy() when modifying DataFrames
2. ALWAYS convert numeric fields before comparison: pd.to_numeric(df['column'], errors='coerce').fillna(0)
3. ALL hour/count fields are STRINGS in the data - you MUST convert them to numeric!
4. Duration is in HOURS, divide by hours_per_day (10) for days
5. For open-ended: tasks NOT in taskpred['pred_task_id'] (no successor)
6. For dangling: tasks NOT in taskpred['task_id'] (no predecessor)
7. Exclude TT_LOE and milestones from most analyses
8. Store final result in variable named 'result'
9. Result must be dict, list, or simple value (JSON serializable)
10. Limit lists to 50 items max

EXAMPLE CODE PATTERNS:

# Get tasks and convert ALL numeric types
tasks = latest['df']['tasks'].copy()
tasks['duration_hrs'] = pd.to_numeric(tasks['target_drtn_hr_cnt'], errors='coerce').fillna(0)
tasks['duration_days'] = tasks['duration_hrs'] / hours_per_day
tasks['float_hrs'] = pd.to_numeric(tasks['total_float_hr_cnt'], errors='coerce').fillna(0)
tasks['complete_pct'] = pd.to_numeric(tasks['phys_complete_pct'], errors='coerce').fillna(0)

# Get relationships and convert lag to numeric
taskpred = latest['df']['taskpred'].copy()
taskpred['lag_hrs'] = pd.to_numeric(taskpred['lag_hr_cnt'], errors='coerce').fillna(0)

# FS relationships with positive lag
fs_with_lag = taskpred[(taskpred['pred_type'] == 'PR_FS') & (taskpred['lag_hrs'] > 0)]

# Negative lags (leads)
negative_lags = taskpred[taskpred['lag_hrs'] < 0]

# Long duration (>30 days)
long_dur = tasks[tasks['duration_days'] > 30]

# Critical activities
critical = tasks[tasks['float_hrs'] <= 0]

# Open-ended (no successor)
has_successor = set(taskpred['pred_task_id'].tolist())
all_ids = set(tasks['task_id'].tolist())
open_ended_ids = all_ids - has_successor
open_ended = tasks[(tasks['task_id'].isin(open_ended_ids)) & (~tasks['task_type'].isin(['TT_LOE', 'TT_Mile', 'TT_FinMile']))]

# Comparing two files
feb = get_update_by_month('feb')
mar = get_update_by_month('mar')
if feb and mar:
    feb_tasks = feb['df']['tasks'].copy()
    mar_tasks = mar['df']['tasks'].copy()
    # comparison logic...

Return ONLY valid Python code. Must set result = ... at the end."""

    def get_response_prompt(self, user_query: str, basic_stats: Dict, code_result: Any, code_success: bool, code_error: str = None) -> str:
        """Generate prompt for final response"""

        context = f"""USER QUESTION: {user_query}

PROJECT OVERVIEW (always available):
- Project: {basic_stats.get('data_source', 'N/A')} (Data Date: {basic_stats.get('data_date', 'N/A')})
- Total Activities: {basic_stats.get('total_activities', 'N/A')}
- Project Period: {basic_stats.get('project_start', 'N/A')} to {basic_stats.get('project_finish', 'N/A')}
- Milestones: {basic_stats.get('milestones', 'N/A')}
- LOE Activities: {basic_stats.get('loe_activities', 'N/A')}

STATUS BREAKDOWN:
- Completed: {basic_stats.get('completed', 'N/A')}
- In Progress: {basic_stats.get('in_progress', 'N/A')}
- Not Started: {basic_stats.get('not_started', 'N/A')}

SCHEDULE HEALTH METRICS:
- Critical Activities: {basic_stats.get('critical_count', 'N/A')} ({basic_stats.get('critical_pct', 'N/A')}%)
- Near-Critical: {basic_stats.get('near_critical_count', 'N/A')}
- Negative Float: {basic_stats.get('negative_float_count', 'N/A')}
- Open-Ended Activities: {basic_stats.get('open_ended_count', 'N/A')}
- Dangling Activities: {basic_stats.get('dangling_count', 'N/A')}
- Long Duration (>30d): {basic_stats.get('long_duration_count', 'N/A')}
- Constrained Activities: {basic_stats.get('constrained_activities', 'N/A')}

RELATIONSHIPS:
- Total: {basic_stats.get('total_relationships', 'N/A')}
- With Lag: {basic_stats.get('relationships_with_lag', 'N/A')}
- Negative Lags: {basic_stats.get('negative_lags', 'N/A')}

RESOURCE LOADING:
- Tasks with Resources: {basic_stats.get('tasks_with_resources', 'N/A')} ({basic_stats.get('resource_loaded_pct', 'N/A')}%)

FILES LOADED:
- Baseline: {basic_stats.get('baseline_name', 'N/A')} ({basic_stats.get('baseline_date', 'N/A')})
- Updates: {basic_stats.get('update_count', 0)} files
"""

        if code_success and code_result:
            context += f"""
SPECIFIC ANALYSIS RESULTS:
{json.dumps(code_result, indent=2, default=str)}
"""
        elif code_error:
            context += f"""
Note: Detailed analysis encountered an issue ({code_error}). Responding based on pre-computed statistics.
"""

        context += """
INSTRUCTIONS:
1. Answer the user's question directly and professionally
2. Include specific numbers from the data
3. Highlight any concerns or issues found
4. Provide recommendations if problems are identified
5. Be concise but thorough
6. Use bullet points and formatting for clarity
7. If the question asks for a list, provide the items found
8. If comparing files, show clear before/after differences"""

        return context
