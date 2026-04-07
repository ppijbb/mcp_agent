"""
Result Reader for Pages

Reads and displays agent results from the standardized result storage.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime



class ResultReader:
    """Reads and displays agent results from standardized result storage.
    
    Provides methods to list available agents, retrieve agent results,
    load specific result files, and get the latest result for an agent.
    """

    def __init__(self, base_results_dir: str = "agent_results"):
        self.base_results_dir = Path(base_results_dir)

    def get_available_agents(self) -> List[str]:
        """
        Get list of agents that have results.

        Returns:
            List of agent names
        """
        if not self.base_results_dir.exists():
            return []

        agents = []
        for item in self.base_results_dir.iterdir():
            if item.is_dir():
                agents.append(item.name)

        return sorted(agents)

    def get_agent_results(self, agent_name: str) -> Dict[str, Any]:
        """
        Get all results for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dict containing available results and metadata
        """
        agent_dir = self.base_results_dir / agent_name
        if not agent_dir.exists():
            return {"agent_name": agent_name, "results": [], "total_count": 0}

        results = []
        for json_file in agent_dir.glob("*.json"):
            if json_file.name.endswith("_metadata.json"):
                continue

            # Try to find corresponding metadata
            metadata_file = json_file.with_name(json_file.stem + "_metadata.json")
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception:
                    pass

            results.append({
                "file_path": str(json_file),
                "filename": json_file.name,
                "size": json_file.stat().st_size,
                "modified": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                "metadata": metadata
            })

        # Sort by modification time (newest first)
        results.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "agent_name": agent_name,
            "results": results,
            "total_count": len(results)
        }

    def load_result(self, file_path: str) -> Optional[Any]:
        """
        Load a specific result file.

        Args:
            file_path: Path to the result file

        Returns:
            The loaded result or None if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load result from {file_path}: {e}")
            return None

    def get_latest_result(self, agent_name: str, result_type: str = "general") -> Optional[Any]:
        """
        Get the latest result for an agent.

        Args:
            agent_name: Name of the agent
            result_type: Type of result

        Returns:
            The latest result or None if not found
        """
        latest_path = self.base_results_dir / agent_name / f"{agent_name}_{result_type}_latest.json"
        if latest_path.exists():
            return self.load_result(str(latest_path))
        return None


class ResultDisplay:
    """
    Displays agent results in Streamlit with various visualization options.
    """

    def __init__(self, result_reader: ResultReader):
        self.reader = result_reader

    def display_agent_selector(self) -> Optional[str]:
        """
        Display agent selection dropdown.

        Returns:
            Selected agent name or None
        """
        agents = self.reader.get_available_agents()

        if not agents:
            st.warning("📭 사용 가능한 agent 결과가 없습니다.")
            st.info("Agent를 실행하여 결과를 생성해주세요.")
            return None

        selected_agent = st.selectbox(
            "🤖 Agent 선택",
            agents,
            help="결과를 확인할 agent를 선택하세요"
        )

        return selected_agent

    def display_result_selector(self, agent_name: str) -> Optional[Dict]:
        """
        Display result selection for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Selected result dict or None
        """
        agent_results = self.reader.get_agent_results(agent_name)

        if not agent_results["results"]:
            st.warning(f"📭 {agent_name}의 결과가 없습니다.")
            return None

        # Create options for selectbox
        options = []
        for result in agent_results["results"]:
            timestamp = datetime.fromisoformat(result["modified"].replace("Z", "+00:00"))
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            options.append(f"{formatted_time} ({result['filename']})")

        selected_option = st.selectbox(
            "📅 결과 선택",
            options,
            help="확인할 결과를 선택하세요"
        )

        if selected_option:
            # Find the corresponding result
            selected_index = options.index(selected_option)
            return agent_results["results"][selected_index]

        return None

    def display_result(self, result_data: Any, metadata: Optional[Dict] = None):
        """
        Display result data with appropriate visualization.

        Args:
            result_data: The result data to display
            metadata: Optional metadata about the result
        """
        if result_data is None:
            st.error("❌ 결과 데이터를 불러올 수 없습니다.")
            return

        # Display metadata if available
        if metadata:
            with st.expander("📋 메타데이터", expanded=False):
                st.json(metadata)

        # Display based on data type
        if isinstance(result_data, dict):
            self._display_dict_result(result_data)
        elif isinstance(result_data, list):
            self._display_list_result(result_data)
        elif isinstance(result_data, str):
            self._display_string_result(result_data)
        elif isinstance(result_data, (int, float)):
            self._display_numeric_result(result_data)
        else:
            st.write("📄 결과 데이터:")
            st.write(result_data)

    def _display_dict_result(self, data: Dict):
        """Display dictionary result."""
        st.subheader("📊 결과 데이터")

        # Check if it's a simple key-value dict
        if all(isinstance(v, (str, int, float, bool)) for v in data.values()):
            # Display as metrics
            cols = st.columns(min(3, len(data)))
            for i, (key, value) in enumerate(data.items()):
                with cols[i % 3]:
                    st.metric(key, value)
        else:
            # Display as JSON
            st.json(data)

    def _display_list_result(self, data: List):
        """Display list result."""
        st.subheader("📋 결과 목록")

        if not data:
            st.info("빈 결과입니다.")
            return

        # Check if it's a list of dictionaries (table-like data)
        if all(isinstance(item, dict) for item in data):
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                # Add download button
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv,
                    file_name=f"result_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                # Try to create visualizations
                self._create_data_visualizations(df)

            except Exception as e:
                st.warning(f"테이블로 변환할 수 없습니다: {e}")
                st.json(data)
        else:
            # Simple list
            for i, item in enumerate(data):
                st.write(f"{i+1}. {item}")

    def _display_string_result(self, data: str):
        """Display string result."""
        st.subheader("📄 결과 텍스트")

        # Check if it's JSON string
        try:
            json_data = json.loads(data)
            st.json(json_data)
        except (json.JSONDecodeError, ValueError):
            # Regular text
            st.text_area("결과 내용", data, height=300)

    def _display_numeric_result(self, data: Union[int, float]):
        """Display numeric result."""
        st.subheader("📊 결과 수치")
        st.metric("결과값", data)

    def _create_data_visualizations(self, df):
        """Create visualizations for dataframe data."""
        if df.empty:
            return

        st.subheader("📈 데이터 시각화")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            # Scatter plot
            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox("X축 선택", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y축 선택", [col for col in numeric_cols if col != x_col], key="scatter_y")

                if x_col and y_col:
                    import plotly.express as px
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Histogram
                hist_col = st.selectbox("히스토그램 컬럼 선택", numeric_cols, key="hist_col")
                if hist_col:
                    import plotly.express as px
                    fig = px.histogram(df, x=hist_col, title=f"{hist_col} 분포")
                    st.plotly_chart(fig, use_container_width=True)

        elif len(numeric_cols) == 1:
            # Single numeric column - histogram
            import plotly.express as px
            fig = px.histogram(df, x=numeric_cols[0], title=f"{numeric_cols[0]} 분포")
            st.plotly_chart(fig, use_container_width=True)

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.subheader("📊 카테고리별 분석")

            cat_col = st.selectbox("카테고리 컬럼 선택", categorical_cols, key="cat_col")
            if cat_col:
                value_counts = df[cat_col].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    st.write("카테고리별 개수:")
                    st.dataframe(value_counts.reset_index().rename(columns={cat_col: '카테고리', 'count': '개수'}))

                with col2:
                    import plotly.express as px
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"{cat_col} 분포")
                    st.plotly_chart(fig, use_container_width=True)

    def display_agent_summary(self, agent_name: str):
        """
        Display summary of all results for an agent.

        Args:
            agent_name: Name of the agent
        """
        agent_results = self.reader.get_agent_results(agent_name)

        st.subheader(f"📊 {agent_name} 결과 요약")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 결과 수", agent_results["total_count"])

        if agent_results["results"]:
            latest_result = agent_results["results"][0]
            latest_time = datetime.fromisoformat(latest_result["modified"].replace("Z", "+00:00"))

            with col2:
                st.metric("최신 결과", latest_time.strftime("%Y-%m-%d %H:%M"))

            with col3:
                total_size = sum(r["size"] for r in agent_results["results"])
                st.metric("총 크기", f"{total_size / 1024:.1f} KB")

        # Results table
        if agent_results["results"]:
            st.subheader("📋 결과 목록")

            # Create dataframe for results
            results_data = []
            for result in agent_results["results"]:
                timestamp = datetime.fromisoformat(result["modified"].replace("Z", "+00:00"))
                results_data.append({
                    "파일명": result["filename"],
                    "수정일시": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "크기 (KB)": f"{result['size'] / 1024:.1f}",
                    "파일경로": result["file_path"]
                })

            import pandas as pd
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)


# Global instances
result_reader = ResultReader()
result_display = ResultDisplay(result_reader)
