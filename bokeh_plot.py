from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
import numpy as np
from PIL import Image
import io
import base64

def create_bokeh_plot(image_paths, x, y, labels):
    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(x=x, y=y, labels=labels))

    # Create a new plot
    p = figure(width=800, height=600, title="Interactive Plot with Hover", tools="")

    # Add scatter plot
    p.scatter('x', 'y', source=source, size=20)

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = """
    <div>
        <div><img src="@image_url" alt="image" style="width: 150px;"/></div>
        <div>@labels</div>
    </div>
    """
    hover.formatters = {'@image_url': 'printf'}
    p.add_tools(hover)

    # Configure the plot
    p.xaxis.axis_label = 'X Axis'
    p.yaxis.axis_label = 'Y Axis'

    # Return the plot
    return p

def get_image_base64(image_path):
    # Convert image to base64
    image = Image.open(image_path)
    buffer = io.BytesIO()
    image.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded_image}"

# Sample data
x = [1, 1, 2, 2]
y = [1, 2, 1, 2]
labels = ["A", "B", "C", "D"]
image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg', 'path_to_image3.jpg', 'path_to_image4.jpg']

image_urls = [get_image_base64(path) for path in image_paths]

# Create the plot
plot = create_bokeh_plot(image_urls, x, y, labels)

# Save the plot
output_file("bokeh_plot.html")
save(plot)
