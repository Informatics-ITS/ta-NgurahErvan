import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf
from IPython.display import clear_output
import streamlit as st
import pandas as pd
import networkx as nx
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components
import scipy
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -- Dataframe --
raw_df3 = pd.read_csv(
    r"D:\KULIAH\TUGAS AKHIR\BUKU TA\streamlit\Data\raw_df3.csv")
expanded_disease_dict = pd.read_pickle(
    r"D:\KULIAH\TUGAS AKHIR\BUKU TA\streamlit\Data\expanded_disease_dict.pkl")
renalfailure_scaling_graph = pd.read_pickle(
    r"D:\KULIAH\TUGAS AKHIR\BUKU TA\streamlit\Data\renalfailure_scaling_graph.pkl")
positive_renalfailure_scaling_graph = pd.read_pickle(
    r"D:\KULIAH\TUGAS AKHIR\BUKU TA\streamlit\Data\positive_renalfailure_scaling_graph.pkl")
penyakit_list = sorted(raw_df3['FKL18A'].unique())

# Streamlit UI
st.title("Klasifikasi Penyakit Kronis")

# -- Input Demografis --
nama = st.text_input("Nama")
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
umur = st.number_input("Masukkan Umur", min_value=0, max_value=120, step=1)

# -- Menampilkan Penyakit yg dipilih --
if "riwayat_penyakit" not in st.session_state:
    st.session_state.riwayat_penyakit = []

selected_penyakit = st.selectbox("Pilih Riwayat Penyakit", penyakit_list)

# -- Konversi FKL18A ke disease_group --
riwayat_list_df = pd.DataFrame(
    st.session_state.riwayat_penyakit, columns=['FKL18A'])
riwayat = raw_df3[['FKL18A', 'disease_group']].drop_duplicates()
hasil_konversi = riwayat_list_df.merge(riwayat, on='FKL18A', how='left')


# -- Membuat Graph Individual --
et = {}
node_template = pd.DataFrame(columns=['Name', 'Weight'])
for g in expanded_disease_dict:
    tmp_df = pd.DataFrame(data=[[g, 0]], columns=['Name', 'Weight'])
    node_template = pd.concat([node_template, tmp_df], ignore_index=True)

et = {}
edge_template = pd.DataFrame(columns=['From', 'To', 'Weight'])
for g in expanded_disease_dict:
    for g2 in expanded_disease_dict:
        tmp_df = pd.DataFrame(data=[[g, g2, 0]], columns=[
                              'From', 'To', 'Weight'])
        edge_template = pd.concat([edge_template, tmp_df], ignore_index=True)


def generate_graph_for_individual(disease_sequence, node_template, edge_template):
    cnts_n = {}
    cnts_e = {}
    last_group = "None"

    # Hitung frekuensi node dan edge berdasarkan urutan
    for d in disease_sequence:
        cnts_n[d] = cnts_n.get(d, 0) + 1

        if last_group != "None":
            edge = last_group + '+' + d
            cnts_e[edge] = cnts_e.get(edge, 0) + 1

        last_group = d

    nodes = node_template.copy()
    edges = edge_template.copy()

    nodes['Weight'] = nodes['Name'].apply(lambda x: cnts_n.get(x, 0))
    edges['Weight'] = edges.apply(lambda x: cnts_e.get(
        x['From'] + '+' + x['To'], 0), axis=1)

    return {'nodes': nodes, 'edges': edges}


# -- Memanggil fungsi untuk membuat graph --
disease_sequence = hasil_konversi['disease_group'].tolist()
result_graph = generate_graph_for_individual(
    disease_sequence=disease_sequence,
    node_template=node_template,
    edge_template=edge_template
)

# -- Visualisasi Jaringan --


def visualize_graph_pyvis(nodes_df, edges_df):
    # Buat graph kosong
    G = nx.DiGraph()

    # Tambahkan nodes
    for _, row in nodes_df.iterrows():
        if row['Weight'] > 0:
            G.add_node(row['Name'], size=row['Weight'] * 10)

    # Tambahkan edges
    for _, row in edges_df.iterrows():
        if row['Weight'] > 0:
            G.add_edge(row['From'], row['To'], weight=row['Weight'])

    # Gunakan Pyvis untuk visualisasi
    net = Network(height="500px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    # Simpan sebagai HTML sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name

    # Tampilkan di Streamlit
    components.html(open(html_path, 'r', encoding='utf-8').read(), height=550)


# -- Node Match Feature --#
def calculate_node_match_single(result_graph, scaling_graph):
    match_score = 0
    total_preval = 0
    node_df = result_graph['nodes']

    for _, row in node_df.iterrows():
        d_name = row['Name']
        weight = row['Weight']
        total_preval += weight

        match = scaling_graph['nodes'][scaling_graph['nodes']
                                       ['Name'] == d_name]
        if not match.empty:
            ref_weight = match['Weight'].iloc[0]
            match_score += weight * ref_weight

    if total_preval == 0:
        return 0.0
    return match_score / total_preval


score_node_match = calculate_node_match_single(
    result_graph, renalfailure_scaling_graph)

# -- edge match score --


def calculate_edge_match_single(result_graph, scaling_graph):
    match_score = 0
    total_preval = 0
    edge_df = result_graph['edges'].copy()

    # Tambahkan kolom Fromto pada graph individu
    edge_df['Fromto'] = edge_df['From'] + edge_df['To']

    # Tambahkan kolom Fromto pada scaling graph jika belum ada
    if 'Fromto' not in scaling_graph['edges'].columns:
        scaling_graph['edges']['Fromto'] = scaling_graph['edges']['From'] + \
            scaling_graph['edges']['To']

    for _, row in edge_df.iterrows():
        d_name = row['Fromto']
        weight = row['Weight']
        total_preval += weight

        # Cari edge yang cocok di scaling graph
        match = scaling_graph['edges'][scaling_graph['edges']
                                       ['Fromto'] == d_name]
        if not match.empty:
            ref_weight = match['Weight'].iloc[0]
            match_score += weight * ref_weight

    if total_preval == 0:
        return 0.0
    return match_score / total_preval


score_edge_match = calculate_edge_match_single(
    result_graph, renalfailure_scaling_graph)

# -- Cluster match score --
# Komunitas Besar


def pandas_to_nx(node_df, edge_df):
    G = nx.DiGraph()
    for i in node_df.index:
        row = node_df.loc[i]
        if row['Weight'] == 0:
            continue
        G.add_node(row['Name'], weight=row['Weight'])
    for i in edge_df.index:
        row = edge_df.loc[i]
        if row['Weight'] == 0:
            continue
        G.add_edge(row['From'], row['To'], weight=row['Weight'])
    return G


G = pandas_to_nx(
    renalfailure_scaling_graph['nodes'], renalfailure_scaling_graph['edges'])

communities = nx.community.louvain_communities(G, seed=1)
communities_dict = {}
for i in range(len(communities)):
    for c in communities[i]:
        communities_dict[c] = i

# Komunitas Positive
positive_G = pandas_to_nx(
    positive_renalfailure_scaling_graph['nodes'], positive_renalfailure_scaling_graph['edges'])
positive_communities = nx.community.louvain_communities(positive_G, seed=1)
positive_communities_dict = {}
for i in range(len(positive_communities)):
    for c in positive_communities[i]:
        positive_communities_dict[c] = i


def calculate_cluster_match_single(result_graph, scaling_graph, communities_dict):
    # Hitung total bobot edge yang berada dalam cluster yang sama di scaling graph
    CN_edge_total = 0
    for _, row in scaling_graph['edges'].iterrows():
        from_node = row['From']
        to_node = row['To']
        if (from_node not in communities_dict) or (to_node not in communities_dict):
            continue
        if communities_dict[from_node] != communities_dict[to_node]:
            continue
        CN_edge_total += row['Weight']

    if CN_edge_total == 0:
        return 0.0

    # Hitung match_score berdasarkan edge dalam cluster yang sama di graph hasil
    match_score = 0
    for _, row in result_graph['edges'].iterrows():
        from_node = row['From']
        to_node = row['To']
        if (from_node not in communities_dict) or (to_node not in communities_dict):
            continue
        if communities_dict[from_node] != communities_dict[to_node]:
            continue
        match_score += row['Weight']

    return match_score / CN_edge_total


score_cluster_match = calculate_cluster_match_single(
    result_graph, renalfailure_scaling_graph, communities_dict)

# -- Node Match Positve --
score_node_match_positive = calculate_node_match_single(
    result_graph, positive_renalfailure_scaling_graph)

# -- Edge Match Positve --
score_edge_match_positive = calculate_edge_match_single(
    result_graph, positive_renalfailure_scaling_graph)

# -- Cluster Match Positve --
score_cluster_match_positive = calculate_cluster_match_single(
    result_graph, positive_renalfailure_scaling_graph, positive_communities_dict)

# -- Menambahkan feature graph individu --


def calc_node_graph_features(result_graph, hasil_konversi):
    g = pandas_to_nx(result_graph['nodes'], result_graph['edges'])
    last_disease = hasil_konversi['disease_group'].iloc[-1]
    try:
        eigenvector_cent = nx.eigenvector_centrality(
            g, max_iter=5000, tol=1e-4)[last_disease]
    except nx.PowerIterationFailedConvergence:
        try:
            eigenvector_cent = nx.eigenvector_centrality(
                g, max_iter=1000, tol=1e-3)[last_disease]
        except nx.PowerIterationFailedConvergence:
            eigenvector_cent = nx.degree_centrality(g)[last_disease]
            print(
                f"Warning: Eigenvector centrality failed for patient , using degree centrality instead")
    return (
        g.degree(last_disease),
        eigenvector_cent,
        nx.closeness_centrality(g)[last_disease],
        nx.clustering(g)[last_disease],
        nx.pagerank(g, max_iter=1000)[last_disease],
    )


g_deg = []
e_c = []
c_c = []
c = []
pr = []
if not hasil_konversi.empty and 'disease_group' in hasil_konversi.columns:
    g_deg_, e_c_, c_c_, c_, pr_ = calc_node_graph_features(
        result_graph, hasil_konversi)
    g_deg.append(g_deg_)
    e_c.append(e_c_)
    c_c.append(c_c_)
    c.append(c_)
    pr.append(pr_)
else:
    g_deg_, e_c_, c_c_, c_, pr_ = [0, 0, 0, 0, 0]
    g_deg.append(g_deg_)
    e_c.append(e_c_)
    c_c.append(c_c_)
    c.append(c_)
    pr.append(pr_)

LDGraphDegree = g_deg
LDEigenvectorCentrality = e_c
LDClosenessCentrality = c_c
LDClusteringCoefficient = c
LDPagerank = pr

# -- Mencari Min dan Max graph individu --


def calc_node_graph_features_min_max_single(result_graph, hasil_konversi):
    res = {
        'LDGraphDegree': {'min': 10000, 'max': -10000, 'min_d': [], 'max_d': []},
        'LDEigenvectorCentrality': {'min': 10000, 'max': -10000, 'min_d': [], 'max_d': []},
        'LDClosenessCentrality': {'min': 10000, 'max': -10000, 'min_d': [], 'max_d': []},
        'LDClusteringCoefficient': {'min': 10000, 'max': -10000, 'min_d': [], 'max_d': []},
        'LDPagerank': {'min': 10000, 'max': -10000, 'min_d': [], 'max_d': []},
    }

    if hasil_konversi.empty or 'disease_group' not in hasil_konversi.columns:
        return res  # return default if no input

    g = pandas_to_nx(result_graph['nodes'], result_graph['edges'])

    # Centrality calculations (with fallback strategies)
    try:
        for strategy in [{'max_iter': 5000, 'tol': 1e-6}, {'max_iter': 3000, 'tol': 1e-4}, {'max_iter': 1000, 'tol': 1e-3}]:
            try:
                eigenvector_cent = nx.eigenvector_centrality(g, **strategy)
                break
            except nx.PowerIterationFailedConvergence:
                eigenvector_cent = None
        if eigenvector_cent is None:
            eigenvector_cent = nx.degree_centrality(g)
    except:
        eigenvector_cent = nx.degree_centrality(g)

    try:
        closeness_cent = nx.closeness_centrality(g)
    except:
        closeness_cent = {n: 0 for n in g.nodes()}

    try:
        clustering_coef = nx.clustering(g)
    except:
        clustering_coef = {n: 0 for n in g.nodes()}

    try:
        for strategy in [{'max_iter': 1000, 'tol': 1e-6}, {'max_iter': 500, 'tol': 1e-4}]:
            try:
                pagerank_scores = nx.pagerank(g, **strategy)
                break
            except nx.PowerIterationFailedConvergence:
                pagerank_scores = None
        if pagerank_scores is None:
            pagerank_scores = nx.degree_centrality(g)
    except:
        pagerank_scores = nx.degree_centrality(g)

    for _, row in hasil_konversi.iterrows():
        d = row['disease_group']
        if d not in g.nodes():
            continue
        try:
            scores_ = {
                'LDGraphDegree': g.degree(d),
                'LDEigenvectorCentrality': eigenvector_cent[d],
                'LDClosenessCentrality': closeness_cent[d],
                'LDClusteringCoefficient': clustering_coef[d],
                'LDPagerank': pagerank_scores[d],
            }
            for key, val in scores_.items():
                if val > res[key]['max']:
                    res[key]['max'] = val
                    res[key]['max_d'] = [d]
                elif val == res[key]['max']:
                    res[key]['max_d'].append(d)

                if val < res[key]['min']:
                    res[key]['min'] = val
                    res[key]['min_d'] = [d]
                elif val == res[key]['min']:
                    res[key]['min_d'].append(d)
        except:
            continue

    return res


gf_stats = calc_node_graph_features_min_max_single(
    result_graph, hasil_konversi)

# -- Menambahkan Fitur Num Visit --
Num_Visits = hasil_konversi['disease_group'].shape[0]
Unique_DGs = hasil_konversi['disease_group'].nunique()

# -- Membuat Feature dari graph individu dengan agregasi dan pembobotan--
G = pandas_to_nx(
    renalfailure_scaling_graph['nodes'], renalfailure_scaling_graph['edges'])


def calculate_all_graph_metrics_dict(g: nx.Graph) -> dict:
    degree_dict = dict(g.degree())
    eigenvector_dict = nx.eigenvector_centrality(g, max_iter=2000)
    closeness_dict = nx.closeness_centrality(g)
    clustering_dict = nx.clustering(g)
    result = {}
    for node in g.nodes:
        result[node] = {
            'Degree': degree_dict[node],
            'Eigenvector': eigenvector_dict[node],
            'Closeness': closeness_dict[node],
            'Clustering': clustering_dict[node],
        }
    return result


graph_metrics = calculate_all_graph_metrics_dict(G)
# Normalisasi Degree untuk semua pasien
total_degree1 = sum(node_data['Degree']
                    for node_data in graph_metrics.values())
for node, metrics in graph_metrics.items():
    metrics['Degree'] = metrics['Degree'] / total_degree1


def calculate_patient_graph_metrics_dict():
    result = {}
    graph = pandas_to_nx(
        renalfailure_scaling_graph['nodes'], renalfailure_scaling_graph['edges'])
    degree_dict = dict(graph.degree())
    eigenvector_dict = nx.eigenvector_centrality(graph, max_iter=5000)
    closeness_dict = nx.closeness_centrality(graph)
    clustering_dict = nx.clustering(graph)
    node_metrics = {}
    for node in graph.nodes:
        node_metrics[node] = {
            'Degree': degree_dict[node] / (len(graph.nodes) - 1) if len(graph.nodes) > 1 else 0,
            'Eigenvector': eigenvector_dict[node],
            'Closeness': closeness_dict[node],
            'Clustering': clustering_dict[node],
        }
    result = node_metrics

    return result


all_patient_metrics = calculate_patient_graph_metrics_dict()
total_degree = sum(node_data['Degree']
                   for node_data in all_patient_metrics.values())
for node, metrics in all_patient_metrics.items():
    metrics['Degree'] = metrics['Degree'] / \
        total_degree if total_degree != 0 else 0

# Menghitung skor akhir untuk setiap pasien
final_patient_scores = {
    'Degree': 0.0,
    'Eigenvector': 0.0,
    'Closeness': 0.0,
    'Clustering': 0.0
}
total_nodes = len(all_patient_metrics)

for node_name, metrics in all_patient_metrics.items():
    if node_name in graph_metrics:
        for centrality in ['Degree', 'Eigenvector', 'Closeness', 'Clustering']:
            final_patient_scores[centrality] += metrics[centrality] * \
                graph_metrics[node_name][centrality]
if total_nodes > 0:
    for centrality in final_patient_scores:
        final_patient_scores[centrality] /= total_nodes

# -- Menambahkan Riwayat Penyakit --


def teks_riwayat(hasil_konversi1):
    hasil_konversi1['FKL18A'] = hasil_konversi1['FKL18A'].str.lower()
    hasil_konversi1['FKL18A'] = hasil_konversi1['FKL18A'].str.replace(
        ',', '', regex=True)
    hasil_konversi1['FKL18A'] = hasil_konversi1['FKL18A'].str.replace(
        r'[-\[\]()\'"]', '', regex=True)
    hasil_riwayat_fkl18a = ', '.join(hasil_konversi1['FKL18A'].astype(str))
    return hasil_riwayat_fkl18a


teks_riwayat_penyakit = teks_riwayat(hasil_konversi1=hasil_konversi)

# -- Menggabungkan Semua Fitur --
summary_dict = {
    "node_match": [score_node_match],
    "edge_match": [score_edge_match],
    "cluster_match": [score_cluster_match],
    "node_match_positive": [score_node_match_positive],
    "edge_match_positive": [score_edge_match_positive],
    "cluster_match_positive": [score_cluster_match_positive],
    "LDGraphDegree": [LDGraphDegree[0]],
    "LDEigenvectorCentrality": [LDEigenvectorCentrality[0]],
    "LDClosenessCentrality": [LDClosenessCentrality[0]],
    "LDClusteringCoefficient": [LDClusteringCoefficient[0]],
    "LDPagerank": [LDPagerank[0]],
    "LDGraphDegree (Max)": [gf_stats['LDGraphDegree']['max']],
    "LDGraphDegree (Min)": [gf_stats['LDGraphDegree']['min']],
    "LDEigenvectorCentrality (Max)": [gf_stats['LDEigenvectorCentrality']['max']],
    "LDEigenvectorCentrality (Min)": [gf_stats['LDEigenvectorCentrality']['min']],
    "LDClosenessCentrality (Max)": [gf_stats['LDClosenessCentrality']['max']],
    "LDClosenessCentrality (Min)": [gf_stats['LDClosenessCentrality']['min']],
    "LDClusteringCoefficient (Max)": [gf_stats['LDClusteringCoefficient']['max']],
    "LDClusteringCoefficient (Min)": [gf_stats['LDClusteringCoefficient']['min']],
    "LDPagerank (Max)": [gf_stats['LDPagerank']['max']],
    "LDPagerank (Min)": [gf_stats['LDPagerank']['min']],
    "Num Visits": [Num_Visits],
    "Unique Disease Groups": [Unique_DGs],
    "Degree": [final_patient_scores['Degree']],
    "Eigenvector": [final_patient_scores['Eigenvector']],
    "Closeness": [final_patient_scores['Closeness']],
    "Clustering": [final_patient_scores['Clustering']],
    "Riwayat Penyakit": [teks_riwayat_penyakit]
}
summary_df_2col = pd.DataFrame({
    "Fitur": list(summary_dict.keys()),
    "Nilai": [v[0] for v in summary_dict.values()]
})

# -- melakukan scaling fitur numerik --
# Pilih baris yang ingin di-scale (bukan Riwayat Penyakit)
summary_df_2col_scaled = summary_df_2col.copy()
mask_numeric = summary_df_2col["Fitur"] != "Riwayat Penyakit"
numeric_values = summary_df_2col.loc[mask_numeric, "Nilai"].astype(
    float).values.reshape(-1, 1)

# Scaling
scaler = StandardScaler()
scaled_values = scaler.fit_transform(numeric_values)

# import tokenizer
with open('Data/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Masukkan hasil scaling ke DataFrame
summary_df_2col_scaled.loc[mask_numeric,
                           "Nilai (Scaled)"] = scaled_values.flatten()
summary_df_2col_scaled.loc[~mask_numeric,
                           "Nilai (Scaled)"] = summary_df_2col_scaled.loc[~mask_numeric, "Nilai"]

# -- Lakukan Embedding kepada Riwayat Penyakit --
riwayat_text = summary_df_2col.loc[summary_df_2col["Fitur"]
                                   == "Riwayat Penyakit", "Nilai"].values[0]
MAX_LEN = 128  # harus sama dengan saat training

# Tokenisasi dan padding
sequence = tokenizer.texts_to_sequences([riwayat_text])
sequence_padded = pad_sequences(
    sequence, maxlen=MAX_LEN, padding='post', truncating='post')

# Menambahkan Attention Layer


class AttentionLayer1(Layer):
    def _init_(self, **kwargs):
        super(AttentionLayer1, self)._init_(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='attention_weights')
        self.b = self.add_weight(shape=(1,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='attention_bias')
        super(AttentionLayer1, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, input_dim)
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) +
                       self.b)  # (batch_size, time_steps, 1)
        alpha = tf.nn.softmax(e, axis=1)  # (batch_size, time_steps, 1)
        context = inputs * alpha  # (batch_size, time_steps, input_dim)
        context_vector = tf.reduce_sum(
            context, axis=1)  # (batch_size, input_dim)
        return context_vector


# -- Loading the model Renal Failure--

model_lstm2 = load_model(
    "Data/model_lstm_unstructured.h5",
    custom_objects={"AttentionLayer1": AttentionLayer1}
)

# -- Melakukan Prediksi Renal Failure --
# Prediksi
prediksi = model_lstm2.predict(sequence_padded)


# --

if st.button("Tambah"):
    st.session_state.riwayat_penyakit.append(selected_penyakit)

st.write("Daftar Riwayat Penyakit yang Dipilih:")
for idx, penyakit in enumerate(st.session_state.riwayat_penyakit, 1):
    st.write(f"{idx}. {penyakit}")

# -- Menampilkan Dataframe Hasil Konversi --
if st.button("Submit"):
    st.success("Data berhasil disimpan!")
    st.write("**Nama:**", nama)
    st.write("**Jenis Kelamin:**", jenis_kelamin)
    st.write("**Umur:**", umur)

    st.subheader("**Riwayat Penyakit:**")
    st.dataframe(hasil_konversi.reset_index(drop=True))
    st.subheader("Graph Nodes")
    st.dataframe(result_graph['nodes'])
    st.subheader("Graph Edges")
    st.dataframe(result_graph['edges'])
    st.subheader("Graph Jaringan Individu")
    visualize_graph_pyvis(result_graph['nodes'], result_graph['edges'])
    st.subheader("Ringkasan Hasil Fitur")
    st.dataframe(summary_df_2col)
    st.subheader("Prediksi Risiko Renal Failure")
    # Tentukan kategori risiko berdasarkan probabilitas
    probability = prediksi[0][0]
    if probability >= 0.7:
        risk_level = "TINGGI"
        risk_color = "#dc3545"  # Merah
        bg_color = "#f8d7da"
        icon = "⚠️"
    elif probability >= 0.4:
        risk_level = "SEDANG"
        risk_color = "#fd7e14"  # Orange
        bg_color = "#fff3cd"
        icon = "⚡"
    else:
        risk_level = "RENDAH"
        risk_color = "#28a745"  # Hijau
        bg_color = "#d4edda"
        icon = "✅"
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {bg_color} 0%, #ffffff 100%);
            padding: 32px 24px;
            border-radius: 18px;
            border-left: 7px solid {risk_color};
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            margin: 30px 0 20px 0;
        ">
            <div style="text-align:center;">
                <h2 style="color:#2c3e50; margin-bottom:8px; font-weight:700;">
                    {icon} Hasil Prediksi Risiko Renal Failure
                </h2>
                <h3 style="color:#007acc; margin-bottom:18px;">
                    Dear <span style="font-weight: bold;">{nama}</span>,
                </h3>
            </div>
            <div style="
                display:flex;
                align-items:center;
                justify-content:center;
                gap:32px;
                margin-bottom:18px;
            ">
                <div style="
                    font-size:18px;
                    color:#2c3e50;
                ">
                    Probabilitas terkena <b>Renal Failure</b>:
                </div>
                <div style="
                    font-size:38px;
                    font-weight: bold;
                    color:{risk_color};
                    background: #fff;
                    border-radius: 12px;
                    padding: 8px 28px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    margin-left:10px;
                ">
                    {probability:.1%}
                </div>
            </div>
            <div style="text-align:center; margin-bottom:18px;">
                <span style="
                    background-color:{risk_color};
                    color:white;
                    padding:8px 28px;
                    border-radius:24px;
                    font-size:18px;
                    font-weight:600;
                    letter-spacing:1px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                ">
                    RISIKO {risk_level}
                </span>
            </div>
            <div style="
                margin-top: 18px;
                padding: 16px 18px;
                background-color: #f8f9fa;
                border-radius: 10px;
                border-left: 5px solid #17a2b8;
            ">
                <h4 style="color: #17a2b8; margin: 0 0 8px 0;">💡 Informasi Penting:</h4>
                <p style="margin: 0; color: #495057; line-height: 1.6;">
                    Hasil ini berdasarkan analisis riwayat penyakit Anda menggunakan Pendekatan 
                    <b>Deep Learning</b>.<br>
                    <b>Konsultasikan dengan dokter</b> untuk penanganan yang tepat.
                </p>
            </div>
            <div style="
                margin-top: 18px;
                text-align: center;
                padding: 12px;
                background-color: #e3f2fd;
                border-radius: 8px;
            ">
                <span style="color: #1976d2; font-weight: 500; font-size:16px;">
                    🏥 <b>Rekomendasi:</b> Lakukan pemeriksaan rutin dan konsultasi dengan dokter spesialis.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
