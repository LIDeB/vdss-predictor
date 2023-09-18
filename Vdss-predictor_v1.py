# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:41:37 2020

@author: Lucas
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from rdkit import Chem
from openbabel import openbabel
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go

# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64


#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='LiADME- VDss predictor', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

st.write("[![Website](https://img.shields.io/badge/website-LIDeB-blue)](https://lideb.biol.unlp.edu.ar)[![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)](https://twitter.com/intent/follow?screen_name=LIDeB_UNLP)")
st.subheader("📌" "About Us")
st.markdown("We are a drug discovery team with an interest in the development of publicly available open-source customizable cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. Our research group is focused on computer-guided drug repurposing and rational discovery of new drug candidates to treat epilepsy and neglected tropical diseases.")


# Introduction
#---------------------------------#

st.title(':computer: _VDss predictor_ ')

st.write("""

**It is a free web-application for Volume of Distribution Prediction**

The volume of distribution at steady state (VDss) is an important pharmacokinetic
parameter that determines, along with clearance, the half-life of a compound. It
is proportional to the quantity of the chemical distributed into body tissues.
 

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [MOLVS](https://molvs.readthedocs.io/), [Openbabel](https://github.com/openbabel/openbabel)

**Workflow:**
""")


image = Image.open('workflow_VDss.png')
st.image(image, caption='VDss predictor workflow')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your SMILES')
st.sidebar.markdown("""
[Example TXT input file](https://raw.githubusercontent.com/LIDeB/OCT1-predictor/main/example_file.txt)        
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a TXT file with one SMILES per line", type=["txt"])


#%% Standarization by MOLVS ####
####---------------------------------------------------------------------------####

def estandarizador(df):
    s = Standardizer()
    moleculas = df[0].tolist()
    moleculas_estandarizadas = []
    i = 1
    t = st.empty()

    for molecula in moleculas:
        try:
            smiles = molecula.strip()
            mol = Chem.MolFromSmiles(smiles)
            standarized_mol = s.super_parent(mol) 
            smiles_estandarizado = Chem.MolToSmiles(standarized_mol)
            moleculas_estandarizadas.append(smiles_estandarizado)
            # st.write(f'\rProcessing molecule {i}/{len(moleculas)}', end='', flush=True)
            t.markdown("Processing molecules: " + str(i) +"/" + str(len(moleculas)))

            i = i + 1
        except:
            moleculas_estandarizadas.append(molecula)
    df['standarized_SMILES'] = moleculas_estandarizadas
    return df


#%% Protonation state at pH 7.4 ####
####---------------------------------------------------------------------------####

def charges_ph(molecule, ph):

    # obConversion it's neccesary for saving the objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    
    # create the OBMol object and read the SMILE
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule)
    
    # Add H, correct pH and add H again, it's the only way it works
    mol.AddHydrogens()
    mol.CorrectForPH(7.4)
    mol.AddHydrogens()
    
    # transforms the OBMOl objecto to string (SMILES)
    optimized = obConversion.WriteString(mol)
    
    return optimized

def smile_obabel_corrector(smiles_ionized):
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize = False)
    
    # checks if the ether group is wrongly protonated
    pattern1 = Chem.MolFromSmarts('[#6]-[#8-]-[#6]')
    if mol1.HasSubstructMatch(pattern1):
        # gets the atom number for the O wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern1)
        at_matches_list = [y[1] for y in at_matches]
        # changes the charged for each O atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    pattern12 = Chem.MolFromSmarts('[#6]-[#8-]-[#16]')
    if mol1.HasSubstructMatch(pattern12):
        # gets the atom number for the O wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern12)
        at_matches_list = [y[1] for y in at_matches]
        # changes the charged for each O atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
            
    # checks if the nitro group is wrongly protonated in the oxygen
    pattern2 = Chem.MolFromSmarts('[#6][O-]=[N+](=O)[O-]')
    if mol1.HasSubstructMatch(pattern2):
        # print('NO 20')
        patt = Chem.MolFromSmiles('[O-]=[N+](=O)[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('O[N+]([O-])=O')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated in the oxygen
    pattern21 = Chem.MolFromSmarts('[#6]-[O-][N+](=O)=[O-]')
    if mol1.HasSubstructMatch(pattern21):
        # print('NO 21')
        patt = Chem.MolFromSmiles('[O-][N+](=O)=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[O][N+](=O)-[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]
        
    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern22 = Chem.MolFromSmarts('[#8-][N+](=[#6])=[O-]')
    if mol1.HasSubstructMatch(pattern22):
        # print('NO 22')
        patt = Chem.MolFromSmiles('[N+]([O-])=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[N+]([O-])-[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern23 = Chem.MolFromSmarts('[#6][N+]([#6])([#8-])=[O-]')
    if mol1.HasSubstructMatch(pattern23):
        # print('NO 23')
        patt = Chem.MolFromSmiles('[N+]([O-])=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[N+]([O-])[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern24 = Chem.MolFromSmarts('[#6]-[#8][N+](=O)=[O-]')
    if mol1.HasSubstructMatch(pattern24):
        # print('NO 24')
        patt = Chem.MolFromSmiles('[O][N+](=O)=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[O][N+](=O)[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the 1H-tetrazole group is wrongly protonated
    pattern3 = Chem.MolFromSmarts('[#7]-1-[#6]=[#7-]-[#7]=[#7]-1')
    if mol1.HasSubstructMatch(pattern3):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern3)
        at_matches_list = [y[2] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    # checks if the 2H-tetrazole group is wrongly protonated
    pattern4 = Chem.MolFromSmarts('[#7]-1-[#7]=[#6]-[#7-]=[#7]-1')
    if mol1.HasSubstructMatch(pattern4):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern4)
        at_matches_list = [y[3] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
        
    # checks if the 2H-tetrazole group is wrongly protonated, different disposition of atoms
    pattern5 = Chem.MolFromSmarts('[#7]-1-[#7]=[#7]-[#6]=[#7-]-1')
    if mol1.HasSubstructMatch(pattern5):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern4)
        at_matches_list = [y[4] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    smile_checked = Chem.MolToSmiles(mol1)
    return smile_checked



#%% formal charge calculation

def formal_charge_calculation(descriptores):
    smiles_list = descriptores["Smiles_OK"]
    charges = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            charge = Chem.rdmolops.GetFormalCharge(mol)
            charges.append(charge)
        except:
            charges.append(None)
        
    descriptores["Formal_charge"] = charges
    return descriptores


#%% Calculating molecular descriptors
### ----------------------- ###

def calcular_descriptores(data):
    
    data1x = pd.DataFrame()
    df_quasi_final_estandarizado = estandarizador(data)
    suppl = list(df_quasi_final_estandarizado["standarized_SMILES"])

    smiles_ph_ok = []
    t = st.empty()

    for i,molecula in enumerate(suppl):
        smiles_ionized = charges_ph(molecula, 7.4)
        smile_checked = smile_obabel_corrector(smiles_ionized)
        smile_final = smile_checked.rstrip()
        smiles_ph_ok.append(smile_final)
        
    df_quasi_final_estandarizado["Final_SMILES"] = smiles_ph_ok
    
    calc = Calculator(descriptors, ignore_3D=True) 
    # t = st.empty()
   
    smiles_ok = []
    for i,smiles in enumerate(smiles_ph_ok):
        if __name__ == "__main__":
                if smiles != None:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        freeze_support()
                        descriptor1 = calc(mol)
                        resu = descriptor1.asdict()
                        solo_nombre = {'NAME' : f'SMILES_{i+1}'}
                        solo_nombre.update(resu)

                        solo_nombre = pd.DataFrame.from_dict(data=solo_nombre,orient="index")
                        data1x = pd.concat([data1x, solo_nombre],axis=1, ignore_index=True)
                        smiles_ok.append(smiles)
                        t.markdown("Calculating descriptors for molecule: " + str(i +1) +"/" + str(len(smiles_ph_ok)))
                    except:
                        
                        st.write(f'\rMolecule {smiles} has been removed (molecule not allowed by Mordred descriptor)')
                else:
                    pass

    data1x = data1x.T
    descriptores = data1x.set_index('NAME',inplace=False).copy()
    descriptores = descriptores.reindex(sorted(descriptores.columns), axis=1)   
    descriptores.replace([np.inf, -np.inf], np.nan, inplace=True)
    descriptores = descriptores.apply(pd.to_numeric, errors = 'coerce') 
    descriptores["Smiles_OK"] = smiles_ok
    descriptors_total = formal_charge_calculation(descriptores)
    
    return descriptors_total, smiles_ok

#%% Determining Applicability Domain (AD)

def applicability_domain(prediction_set_descriptors, descriptors_model):
    
    descr_training = pd.read_csv("models/" + "training_set_vdss.csv")
    desc = descr_training[descriptors_model]
    t_transpuesto = desc.T
    multi = t_transpuesto.dot(desc)
    inversa = np.linalg.inv(multi)
    
    # Luego la base de testeo
    desc_sv = prediction_set_descriptors.copy()
    sv_transpuesto = desc_sv.T
    
    multi1 = desc_sv.dot(inversa)
    sv_transpuesto.reset_index(drop=True, inplace=True) 
    multi2 = multi1.dot(sv_transpuesto)
    diagonal = np.diag(multi2)
    
    # valor de corte para determinar si entra o no en el DA
    
    h2 = 2*(desc.shape[1]/desc.shape[0])  ## El h es 2 x Num de descriptores dividido el Num compuestos training. Mas estricto
    h3 = 3*(desc.shape[1]/desc.shape[0])  ##  Mas flexible
    
    diagonal_comparacion = list(diagonal)
    resultado_palanca2 =[]
    for valor in diagonal_comparacion:
        if valor < h2:
            resultado_palanca2.append(True)
        else:
            resultado_palanca2.append(False)
    resultado_palanca3 =[]
    for valor in diagonal_comparacion:
        if valor < h3:
            resultado_palanca3.append(True)
        else:
            resultado_palanca3.append(False)         
    return resultado_palanca2, resultado_palanca3


#%% Removing molecules with na in any descriptor

def all_correct_model(descriptors_total,loaded_desc, smiles_list):
    
    total_desc = loaded_desc.copy()
    X_final = descriptors_total[total_desc]
    X_final["SMILES_OK"] = smiles_list
    rows_with_na = X_final[X_final.isna().any(axis=1)]         # Find rows with NaN values
    for molecule in rows_with_na["SMILES_OK"]:
        st.write(f'\rMolecule {molecule} has been removed (NA value  in any of the necessary descriptors)')
    X_final1 = X_final.dropna(axis=0,how="any",inplace=False)
    
    smiles_final = X_final1["SMILES_OK"]
    return X_final1, smiles_final

 # Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Substrate":
        return 'lightgreen'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        return 'red'


#%% Predictions        

def predictions(loaded_model, loaded_desc, X_final1):
    i = 0
    
    estimator = loaded_model
    descriptors_model = loaded_desc.copy()
    
    X = X_final1[descriptors_model]
    score_predictions = estimator.predict(X)
    score_predictions = 10**score_predictions
    #st.write(score_predictions)
    resultado_palanca2, resultado_palanca3  = applicability_domain(X, descriptors_model)
    i = i + 1 
    
    dataframe_scores = pd.DataFrame(score_predictions)
    dataframe_scores.index = smiles_final
    
    palancas_final2 = pd.DataFrame(resultado_palanca2)
    palancas_final2.index = smiles_final
    palancas_final2.rename(columns={0: "Confidence"},inplace=True)
    
    palancas_final3 = pd.DataFrame(resultado_palanca3)
    palancas_final3.index = smiles_final
    palancas_final3.rename(columns={0: "DA3"},inplace=True)
    
    final_file = pd.concat([dataframe_scores,palancas_final2, palancas_final3], axis=1)
    
    final_file.rename(columns={0: "Prediction"},inplace=True)
    
    final_file.loc[final_file["Confidence"] == True, 'Confidence'] = 'HIGH'
    final_file.loc[(final_file["DA3"] == True) & (final_file["Confidence"] != "HIGH"), 'Confidence'] = 'MEDIUM'
    final_file.loc[final_file["DA3"] == False, 'Confidence'] = 'LOW'

    final_file.drop(columns=['DA3'],inplace=True)
            
    df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
    styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
    return final_file, styled_df



#%% Create plot:



def final_plot(final_file):
    non_conclusives = len(final_file[final_file['Confidence'] == "LOW"]) 
    high_vdss = len(final_file[(final_file['Confidence'] != "LOW") & (final_file['Prediction'] >= 10)])
    medium_vdss = len(final_file[(final_file['Confidence'] != "LOW") & (final_file['Prediction'] > 1) & (final_file['Prediction'] < 10)])
    low_vdss = len(final_file[(final_file['Confidence'] != "LOW") & (final_file['Prediction'] < 1)])
    
    keys = ["High VDss (>= 10)", "Medium Vdss (1 < VDss < 10)", "Low VDss (< 1)", "Non conclusive"]
    fig = go.Figure(go.Pie(labels=keys, values=[high_vdss, medium_vdss, low_vdss, non_conclusives]))
    
    fig.update_layout(title_text=None)
    
    return fig


#%%
def filedownload1(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="VDss_prediction_results.csv">Download CSV File with results</a>'
    return href

#%% CORRIDA

loaded_model = pickle.load(open("models/" + "Vdss_model.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "Vdss_descriptors.pickle", 'rb'))


if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        data = pd.read_csv(uploaded_file_1,sep="\t",header=None)       
        descriptors_total, smiles_list = calcular_descriptores(data)
        X_final1, smiles_final = all_correct_model(descriptors_total,loaded_desc, smiles_list)
        final_file, styled_df = predictions(loaded_model, loaded_desc, X_final1)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)

        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)
       

# Example file
else:
    st.info('👈🏼👈🏼👈🏼      Awaiting for TXT file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        data = pd.read_csv("example_file.txt",sep="\t",header=None)
        descriptors_total, smiles_list = calcular_descriptores(data)
        X_final1, smiles_final = all_correct_model(descriptors_total,loaded_desc, smiles_list)
        final_file, styled_df = predictions(loaded_model, loaded_desc, X_final1)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)
        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
  
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display: ;
 text-align: center' href="https://twitter.com/capigol" target="_blank">Lucas Alberca</a> and <a style='display: ; 
 text-align: center' href="https://twitter.com/carobellera" target="_blank">Caro Bellera</a> for <a style='display: ; 
 text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

