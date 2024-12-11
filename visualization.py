import plotly.graph_objects as go


def visualize_results(metrics):
    """
    Create a visualization of metrics using Plotly.

    Args:
        metrics (dict): Dictionary of evaluation metrics.

    Returns:
        Plotly Figure: A bar chart of the metrics.
    """
    fig = go.Figure()
    for metric_name, metric_value in metrics.items():
        fig.add_trace(go.Bar(name=metric_name, x=[
                      metric_name], y=[metric_value]))
    fig.update_layout(title="Evaluation Metrics", barmode='group')
    return fig
