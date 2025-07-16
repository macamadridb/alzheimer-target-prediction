# este codifo contiene las funciones relacionadas con la carga, mapeo y preprocesameinto de mis archivos en el formato requerido por PyTorch Geometric.
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from collections import defaultdict

def load_files(edge_path, go_path, protein_metadata_path, go_metadata_path):

    edges_df = pd.read_csv(edge_path, sep='\t')
    go_terms_df = pd.read_csv(go_path, sep='\t')
    protein_metadata_df = pd.read_csv(protein_metadata_path, sep=',')
    go_metadata_df = pd.read_csv(go_metadata_path, sep=',')

    return edges_df, go_terms_df, protein_metadata_df, go_metadata_df

def create_node_mappings(edges_df, go_terms_df, protein_metadata_df):

# Crea un mapeo único de IDs de proteína a índices numéricos, necesario para construir el grafo y las matrices de características.
# Parámetros:
# - edges_df: DataFrame con las columnas 'proteina1' y 'proteina2'.
# - go_terms_df: DataFrame con las columnas 'proteina' y 'GO_term'.
# - protein_metadata_df: DataFrame con columna 'proteina'.

# Retorna:
# - protein_to_idx: Diccionario {proteína: índice}.
# - idx_to_protein: Diccionario {índice: proteína}.
# - all_proteins: Lista de todas las proteínas únicas.

# Concatena todas las proteínas de los DataFrames relevantes
    all_proteins_raw = pd.concat([
        edges_df['proteina1'],
        edges_df['proteina2'], 
        go_terms_df['proteina'],
        protein_metadata_df['proteina']
    ]).unique()
    
# Filtrar posibles IDs no válidos o términos GO que se cuelen en la lista de proteínas
    all_proteins = [
        p for p in all_proteins_raw 
        if isinstance(p, str) and not p.startswith('GO:') and pd.notna(p)
        ]

# Crear mapeos de proteínas a índices y viceversa
    protein_to_idx = {protein: i for i, protein in enumerate(all_proteins)}
    idx_to_protein = {i: protein for protein, i in protein_to_idx.items()}

    return protein_to_idx, idx_to_protein, all_proteins

def get_go_ontology_mapping(go_metadata_df):
# Crea un mapeo de GO_term a su ontología (BP, MF, CC).
# Parámetro:
# - go_metadata_df: DataFrame con columnas 'GO_term' y 'Ontology'.
# Retorna:
# - go_ontology_map: Diccionario que mapea términos GO a su ontología (BP, MF, CC).
# - go_terms_by_ontology: {'BP': [...], 'MF': [...], 'CC': [...], 'unknown': [...]}

    go_ontology_map = {}
    go_terms_by_ontology = {'BP': [], 'MF': [], 'CC': [], 'unknown': []}
    for _, row in go_metadata_df.iterrows():
        term = row['GO_term']
        ontology_info = str(row['Ontology']).lower() # Asegura que siempre sea string y minúscula 
        
        if 'biological process' in ontology_info:
            go_ontology_map[term] = 'BP'
            go_terms_by_ontology['BP'].append(term)
        elif 'molecular function' in ontology_info:
            go_ontology_map[term] = 'MF'
            go_terms_by_ontology['MF'].append(term)
        elif 'cellular component' in ontology_info:
            go_ontology_map[term] = 'CC'
            go_terms_by_ontology['CC'].append(term)
        else:
            go_ontology_map[term] = 'unknown'
            go_terms_by_ontology['unknown'].append(term)

# Crear la opción 'all': unión de BP, MF y CC
    go_terms_by_ontology['all'] = (
        go_terms_by_ontology['BP'] + 
        go_terms_by_ontology['MF'] + 
        go_terms_by_ontology['CC']
    )

    return go_ontology_map, go_terms_by_ontology

def create_node_features(protein_to_idx, go_terms_df, protein_metadata_df, go_metadata_df, go_ontology_filter='all'):

#Crea las características de los nodos (x) para la GNN.
#Incluye la opción de filtrar términos GO por ontología.

    num_nodes = len(protein_to_idx) 
    
    # Procesar metadata_proteina.csv
    protein_features = pd.DataFrame(index=protein_to_idx.keys())
    protein_features = protein_features.merge(protein_metadata_df.set_index('proteina'), 
                                              left_index=True, right_index=True, how='left')
    
    # Crea DataFrame con las proteínas del grafo como indice y se le añaden las columnas de metadata
    protein_features['Target_type'] = protein_features['Target_type'].fillna('unknown')
    protein_features['Target_group'] = protein_features['Target_group'].fillna('')
    protein_features['Target_group_score_normalized'] = protein_features['Target_group_score_normalized'].fillna(0.0)
    protein_features['DEG'] = protein_features['DEG'].fillna('none')

    # Codificación de 'target_type', este es 'single' o 'complex'
    le_target_type = LabelEncoder()
    protein_features['Target_type_encoded'] = le_target_type.fit_transform(protein_features['Target_type'])
    
    # Codificación de 'DEG'
    le_deg = LabelEncoder()
    protein_features['DEG_encoded'] = le_deg.fit_transform(protein_features['DEG'])

    # Codificación de 'target_group' (multi-hot encoding)
    all_target_groups = set()
    for groups in protein_features['Target_group'].dropna():
        for g in str(groups).split(','): # Convertir a string y dividir por comas
            g = g.strip()
            if g: # Asegurarse de que no esté vacío
                all_target_groups.add(g)
    
    mlb_target_group = MultiLabelBinarizer(classes=sorted(list(all_target_groups)))
    # Aplicar MultiLabelBinarizer, manejando casos de valores vacíos
    target_group_encoded = mlb_target_group.fit_transform(
        protein_features['Target_group'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    )
    target_group_df = pd.DataFrame(target_group_encoded, index=protein_features.index, columns=mlb_target_group.classes_)


    # Normalizar 'target_group_score_normalized'
    scaler = StandardScaler()
    protein_features['Target_group_score_normalized_scaled'] = scaler.fit_transform(
        protein_features[['Target_group_score_normalized']]
        )

    # 2. Procesar go.csv y metadata_go.csv para términos GO
    go_ontology_map, go_terms_by_ontology = get_go_ontology_mapping(go_metadata_df)
    
    # Obtener términos GO perfimitos según el filtro de ontología
    valid_go_terms = set(go_terms_by_ontology.get(go_ontology_filter, []))

    # Filtrar el DataFrame de GO_terms 
    filtered_go_terms = go_terms_df[
        (go_terms_df['GO_term'].isin(valid_go_terms)) &
        (go_terms_df['proteina'].isin(protein_to_idx.keys()))
    ]

    # Esto crea una lista de listas, donde cada sublista contiene los términos GO asociados a cada proteína
    # De la forma protein_go_map = {'P1': ['GO:0001', 'GO:0002'], 'P2': ['GO:0003']}
    protein_go_map = defaultdict(list)
    for _, row in filtered_go_terms.iterrows():
        protein_go_map[row['proteina']].append(row['GO_term'])

    # Lista de listas ( para MultiLabelBinarizer)
    go_terms_for_binarizer = [protein_go_map[p] for p in protein_features.index]

    unique_go_terms = sorted(valid_go_terms.intersection(filtered_go_terms['GO_term'].unique()))
    num_nodes_covered_by_go = len({p for p in filtered_go_terms['proteina']})
    num_go_terms_covered = len(unique_go_terms)

    if unique_go_terms:
        mlb_go = MultiLabelBinarizer(classes=unique_go_terms)
        go_features_encoded = mlb_go.fit_transform(go_terms_for_binarizer)
        go_features_df = pd.DataFrame(go_features_encoded,
                                      index = protein_features.index,
                                      columns=[f"GO_{c}" for c in mlb_go.classes_])
    else:
        mlb_go = MultiLabelBinarizer()
        go_features_df = pd.DataFrame(index=protein_features.index)  # DataFrame vacío si no hay términos GO

    # Combinar todas las características en un único DataFrame
    all_features_df = pd.concat([
        protein_features[['Target_type_encoded', 'DEG_encoded', 'Target_group_score_normalized_scaled']],
        target_group_df,
        go_features_df
    ], axis=1)

    # Convertir a tensor de PyTorch, asegurando el orden de los nodos
    X = torch.tensor(all_features_df.loc[list(protein_to_idx.keys())].values, dtype=torch.float)
    
    return X, num_nodes_covered_by_go, num_go_terms_covered, le_target_type, le_deg, mlb_target_group, mlb_go

def create_edge_index_and_attributes(edges_df, protein_to_idx):

    #Crea el edge_index y edge_attr para la GNN.
    
    # Filtrar edges donde ambas proteínas existen en el mapeo
    filtered_edges = edges_df[
        (edges_df['proteina1'].isin(protein_to_idx.keys())) & 
        (edges_df['proteina2'].isin(protein_to_idx.keys()))
    ].copy() 
    
    # Convertir IDs de proteínas a índices numéricos
    src = [protein_to_idx[p] for p in filtered_edges['proteina1']] # src es la columna de origen
    dst = [protein_to_idx[p] for p in filtered_edges['proteina2']] # dst es la columna de destino
    
    # Crear edge_index bidireccional de tamaño (2, num_edges * 2) con los indices numericos de los nodos conectados
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    # Replicar interaction_Score para ambos sentidos del edge
    # Tensor con los pesos de cada arista, duplicado para bidireccionalidad
    edge_attr = torch.tensor(
        filtered_edges['interaction_score'].values.tolist() * 2,
        dtype=torch.float
    ).unsqueeze(1)
    # Transforma de [4] a [4, 1] para que cada peso sea un vector de una dimensión
    
    num_edges_original = len(filtered_edges)
    num_edges_bidirectional = num_edges_original * 2
    
    return edge_index, edge_attr, num_edges_original, num_edges_bidirectional