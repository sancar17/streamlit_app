import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import zipfile

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False,
        max_size=1000000000  # 1GB max size
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename):
    try:
        if contents is None:
            return f"No contents received for file: {filename}"

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(decoded)) as z:
                file_list = z.namelist()
                return html.Div([
                    html.H5(f'Zip file: {filename}'),
                    html.H6(f'Number of files: {len(file_list)}'),
                    html.Ul([html.Li(file) for file in file_list[:10]]),
                    html.P('...' if len(file_list) > 10 else '')
                ])
        else:
            return html.Div([
                html.H5(f'File: {filename}'),
                html.Pre(decoded[:100].decode('utf-8') + '...')
            ])
    except Exception as e:
        return html.Div([
            html.H5(f'Error processing file: {filename}'),
            html.Pre(str(e))
        ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(contents, filename, last_modified):
    if contents is not None:
        children = [
            parse_contents(contents, filename),
            html.Hr(),
            html.Div(f'Last Modified: {last_modified}')
        ]
        return children
    return html.Div('No file uploaded yet.')

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)