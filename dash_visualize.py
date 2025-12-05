#!/usr/bin/env python
"""
TARLの点群データとセグメントラベルを可視化するためのDashアプリケーション
"""

import os
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import argparse
import sys
import colorsys

def generate_colors(n):
    """
    区別しやすいn個の色を生成する
    """
    colors = []
    for i in range(n):
        h = i / n
        s = 0.8
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([r, g, b])
    return np.array(colors)

def load_data(debug_dir, timestamp=None):
    """
    指定したディレクトリからデータを読み込む
    """
    # タイムスタンプが指定されていない場合は最新のものを使用
    if timestamp is None:
        files = os.listdir(debug_dir)
        timestamps = set()
        for f in files:
            if f.startswith('points_q_') and f.endswith('.npy'):
                ts = f.replace('points_q_', '').replace('.npy', '')
                timestamps.add(ts)
        
        if not timestamps:
            print("データが見つかりません。")
            return None
        
        timestamp = sorted(timestamps)[-1]  # 最新のタイムスタンプ
    
    # ファイルパスを構築
    points_q_file = os.path.join(debug_dir, f'points_q_{timestamp}.npy')
    segs_q_file = os.path.join(debug_dir, f'segs_q_{timestamp}.npy')
    points_k_file = os.path.join(debug_dir, f'points_k_{timestamp}.npy')
    segs_k_file = os.path.join(debug_dir, f'segs_k_{timestamp}.npy')
    
    # ファイルの存在確認
    for f in [points_q_file, segs_q_file, points_k_file, segs_k_file]:
        if not os.path.exists(f):
            print(f"ファイル {f} が見つかりません。")
            return None
    
    # データの読み込み
    points_q = np.load(points_q_file)
    segs_q = np.load(segs_q_file)
    points_k = np.load(points_k_file)
    segs_k = np.load(segs_k_file)
    
    print(f"データを読み込みました: {timestamp}")
    print(f"クエリ点群: {points_q.shape}, セグメント: {np.unique(segs_q).shape}")
    print(f"キー点群: {points_k.shape}, セグメント: {np.unique(segs_k).shape}")
    
    return points_q, segs_q, points_k, segs_k, timestamp

def get_timestamps(debug_dir):
    """
    利用可能なタイムスタンプを取得する
    """
    files = os.listdir(debug_dir)
    timestamps = set()
    for f in files:
        if f.startswith('points_q_') and f.endswith('.npy'):
            ts = f.replace('points_q_', '').replace('.npy', '')
            timestamps.add(ts)
    return sorted(timestamps, reverse=True)  # 新しい順

def create_point_cloud_figure(points, segs, title, colorscale=None, highlighted_segs=None):
    """
    点群データからPlotlyの図を作成
    
    Args:
        points: 点群の座標
        segs: セグメントラベル
        title: 図のタイトル
        colorscale: カラースケール（Noneの場合は自動生成）
        highlighted_segs: 強調表示するセグメントIDのリスト（Noneの場合はすべて通常表示）
    """
    # ユニークなセグメントの取得
    unique_segs = np.unique(segs)
    n_segs = len(unique_segs)
    
    # カスタムカラースケールの生成
    if colorscale is None:
        colors = generate_colors(n_segs)
        colorscale = []
        for i, seg_id in enumerate(unique_segs):
            colorscale.append([i/n_segs, f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})'])
            colorscale.append([(i+1)/n_segs, f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})'])
    
    # 特定セグメントの強調表示
    if highlighted_segs is not None and len(highlighted_segs) > 0:
        # カラーマップを作成
        color_array = np.zeros((points.shape[0], 4))  # RGBA形式
        
        # すべての点を灰色にする
        color_array[:, :3] = 0.7  # RGB=0.7（灰色）
        color_array[:, 3] = 0.3   # アルファ=0.3（やや透明）
        
        # 強調表示するセグメントだけカラフルに
        for i, seg_id in enumerate(highlighted_segs):
            mask = segs == seg_id
            # 強調セグメントのインデックスを取得
            if seg_id in unique_segs:
                idx = np.where(unique_segs == seg_id)[0][0]
                rgb = np.array([float(colorscale[idx*2][1].split('(')[1].split(')')[0].split(',')[j]) for j in range(3)]) / 255.0
                color_array[mask, :3] = rgb
                color_array[mask, 3] = 1.0  # 不透明
        
        # 点群のプロット
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_array,
                ),
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>セグメント: %{customdata}<extra></extra>',
                customdata=segs
            )
        ])
    else:
        # 通常の点群プロット
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=segs,
                    colorscale=colorscale,
                    colorbar=dict(
                        title='セグメントID',
                        tickvals=unique_segs,
                        ticktext=[f'ID: {int(seg)}' for seg in unique_segs],
                        lenmode='fraction',
                        len=0.75
                    )
                ),
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>セグメント: %{marker.color}<extra></extra>'
            )
        ])
    
    # レイアウトの設定
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        uirevision='same_camera'  # カメラ設定を維持するための設定
    )
    
    return fig

# Dashアプリケーションの起動
def run_dashboard(debug_dir, port=8051):
    app = dash.Dash(__name__)
    
    timestamps = get_timestamps(debug_dir)
    
    if not timestamps:
        print("エラー: データが見つかりません。")
        return
    
    # すべてのタイムスタンプのデータからセグメントIDを収集
    print("すべてのタイムスタンプからセグメントIDを収集しています...")
    all_segs = set()
    
    for ts in timestamps:
        data = load_data(debug_dir, ts)
        if data is not None:
            points_q, segs_q, points_k, segs_k, _ = data
            all_segs.update(np.unique(segs_q).astype(int))
            all_segs.update(np.unique(segs_k).astype(int))
    
    all_segs = sorted(list(all_segs))
    n_segs = len(all_segs)
    print(f"合計 {n_segs} 個のユニークなセグメントIDを検出しました")
    
    # 最初のデータセットを読み込む
    initial_data = load_data(debug_dir, timestamps[0])
    if initial_data is None:
        print("エラー: データの読み込みに失敗しました。")
        return
    
    # 統一カラースケールの作成
    colors = generate_colors(n_segs)
    colorscale = []
    for i, seg_id in enumerate(all_segs):
        colorscale.append([i/n_segs, f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})'])
        colorscale.append([(i+1)/n_segs, f'rgb({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255})'])
    
    app.layout = html.Div([
        html.H1("TARL点群セグメンテーション可視化", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.Label("タイムスタンプ選択:"),
                dcc.Dropdown(
                    id='timestamp-dropdown',
                    options=[{'label': ts, 'value': ts} for ts in timestamps],
                    value=timestamps[0] if timestamps else None,
                    style={'width': '100%'}
                ),
            ], style={'width': '50%'}),
            
            html.Div([
                html.Label("視点同期:"),
                dcc.RadioItems(
                    id='sync-camera',
                    options=[
                        {'label': 'オン', 'value': 'on'},
                        {'label': 'オフ', 'value': 'off'}
                    ],
                    value='off',  # デフォルトでオフに設定
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '10px'}),
        ], style={'display': 'flex', 'margin': '10px'}),
        
        html.Div([
            html.Label("強調表示するセグメント:"),
            dcc.Checklist(
                id='highlight-segments',
                options=[{'label': f'ID: {int(seg)}', 'value': int(seg)} for seg in all_segs],
                value=[],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                style={'maxHeight': '100px', 'overflow': 'auto'}
            ),
        ], style={'margin': '10px'}),
        
        html.Div([
            html.Button('すべて選択', id='select-all-btn', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('すべて解除', id='clear-all-btn', n_clicks=0),
        ], style={'margin': '10px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='point-cloud-query', style={'height': '80vh'})
            ], style={'width': '50%'}),
            
            html.Div([
                dcc.Graph(id='point-cloud-key', style={'height': '80vh'})
            ], style={'width': '50%'})
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
    
    @app.callback(
        [Output('point-cloud-query', 'figure'),
         Output('point-cloud-key', 'figure')],
        [Input('timestamp-dropdown', 'value'),
         Input('highlight-segments', 'value'),
         Input('sync-camera', 'value')],
        [State('point-cloud-query', 'relayoutData'),
         State('point-cloud-key', 'relayoutData')]
    )
    def update_graphs(timestamp, highlighted_segs, sync_camera, query_layout, key_layout):
        if not timestamp:
            return {}, {}
        
        data = load_data(debug_dir, timestamp)
        if data is None:
            return {}, {}
            
        points_q, segs_q, points_k, segs_k, _ = data
        
        # クエリ点群とキー点群の図を作成
        fig_query = create_point_cloud_figure(points_q, segs_q, 'クエリ点群', colorscale, highlighted_segs)
        fig_key = create_point_cloud_figure(points_k, segs_k, 'キー点群', colorscale, highlighted_segs)
        
        # カメラの同期処理
        if sync_camera == 'on':
            # 共通のシーンIDを設定
            fig_query.layout.scene.uirevision = 'same_camera'
            fig_key.layout.scene.uirevision = 'same_camera'
            
            # 一方のカメラ位置が変更されていれば、もう一方に適用
            if query_layout and 'scene.camera' in query_layout:
                fig_key.layout.scene.camera = query_layout['scene.camera']
            elif key_layout and 'scene.camera' in key_layout:
                fig_query.layout.scene.camera = key_layout['scene.camera']
        else:
            # 各グラフ固有のシーンIDを設定（同期しない）
            fig_query.layout.scene.uirevision = 'query_camera'
            fig_key.layout.scene.uirevision = 'key_camera'
        
        return fig_query, fig_key
    
    @app.callback(
        Output('highlight-segments', 'value'),
        [Input('select-all-btn', 'n_clicks'),
         Input('clear-all-btn', 'n_clicks')],
        [State('highlight-segments', 'options'),
         State('highlight-segments', 'value')]
    )
    def update_segment_selection(select_clicks, clear_clicks, options, current_values):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_values
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'select-all-btn':
            return [option['value'] for option in options]
        elif button_id == 'clear-all-btn':
            return []
        
        return current_values
    
    print(f"Dashサーバーを起動しています（ポート: {port}）...")
    print(f"ブラウザで http://localhost:{port} にアクセスしてください")
    print("※ リモートサーバーで実行している場合は、SSH ポートフォワーディングを設定してください")
    print(f"  ローカル側: ssh -L {port}:localhost:{port} itachi")
    
    # サーバーを起動
    app.run_server(debug=True, host='0.0.0.0', port=port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TARLデータ可視化Webアプリ')
    parser.add_argument('--debug_dir', type=str, required=True, help='デバッグデータのディレクトリ')
    parser.add_argument('--port', type=int, default=8051, help='Webサーバーのポート番号')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.debug_dir):
        print(f"エラー: ディレクトリが見つかりません: {args.debug_dir}")
        sys.exit(1)
    
    run_dashboard(args.debug_dir, args.port) 