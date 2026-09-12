import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Previsão de Vendas",
    page_icon="📊",
    layout="wide"
)

tf.random.set_seed(42)
np.random.seed(42)


# ============================================================
# DATASET
# ============================================================

dados_vendas = {
    "mes": [
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12
    ],
    "investimento_marketing": [
        1000, 1200, 1500, 1800, 2000, 2300,
        2500, 2700, 3000, 3200, 3500, 4000
    ],
    "numero_clientes": [
        100, 120, 135, 150, 170, 190,
        210, 230, 250, 270, 300, 330
    ],
    "vendas": [
        15000, 17000, 19000, 21000, 24000, 27000,
        30000, 33000, 36000, 39000, 43000, 47000
    ]
}


# ============================================================
# DATAFRAME
# ============================================================

def criar_dataframe():
    """Converte o dicionário em DataFrame."""

    try:
        df = pd.DataFrame(dados_vendas)

        if df.empty:
            raise ValueError("O dataset está vazio.")

        colunas_obrigatorias = [
            "mes",
            "investimento_marketing",
            "numero_clientes",
            "vendas"
        ]

        for coluna in colunas_obrigatorias:
            if coluna not in df.columns:
                raise ValueError(
                    f"A coluna '{coluna}' não existe no dataset."
                )

        if df.isnull().values.any():
            raise ValueError(
                "O dataset contém valores vazios."
            )

        return df

    except Exception as erro:
        st.error(
            f"Erro ao carregar o dataset: {erro}"
        )
        return None


# ============================================================
# ANÁLISE BÁSICA
# ============================================================

def analisar_dados(df):
    """Exibe análise básica do dataset."""

    st.subheader("📊 Análise básica")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Registros",
        len(df)
    )

    col2.metric(
        "Colunas",
        len(df.columns)
    )

    col3.metric(
        "Média das vendas",
        f"R$ {df['vendas'].mean():,.2f}"
    )

    col4.metric(
        "Maior venda",
        f"R$ {df['vendas'].max():,.2f}"
    )

    st.write("### Tipos de dados")

    st.dataframe(
        df.dtypes.astype(str).to_frame("Tipo"),
        use_container_width=True
    )

    st.write("### Estatísticas")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.write("### Vendas por mês")

    grafico = df.set_index("mes")["vendas"]

    st.line_chart(
        grafico
    )


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar_dados(df):

    caracteristicas = [
        "mes",
        "investimento_marketing",
        "numero_clientes"
    ]

    X = df[caracteristicas].values.astype(
        np.float32
    )

    y = df["vendas"].values.astype(
        np.float32
    )

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    diferenca = X_max - X_min

    diferenca[diferenca == 0] = 1

    X_normalizado = (
        X - X_min
    ) / diferenca

    y_min = y.min()
    y_max = y.max()

    if y_max == y_min:
        y_max = y_min + 1

    y_normalizado = (
        y - y_min
    ) / (y_max - y_min)

    return (
        X_normalizado,
        y_normalizado,
        X_min,
        X_max,
        y_min,
        y_max
    )


# ============================================================
# MODELO
# ============================================================

def criar_modelo():

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(3,)
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu"
            ),

            tf.keras.layers.Dense(
                1,
                activation="linear"
            )
        ]
    )

    modelo.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return modelo


# ============================================================
# TREINAMENTO
# ============================================================

@st.cache_resource
def treinar_modelo(df):

    (
        X,
        y,
        X_min,
        X_max,
        y_min,
        y_max
    ) = normalizar_dados(df)

    modelo = criar_modelo()

    modelo.fit(
        X,
        y,
        epochs=500,
        verbose=0
    )

    return (
        modelo,
        X_min,
        X_max,
        y_min,
        y_max
    )


# ============================================================
# PREVISÃO
# ============================================================

def realizar_previsao(
    modelo,
    mes,
    marketing,
    clientes,
    X_min,
    X_max,
    y_min,
    y_max
):

    entrada = np.array(
        [[
            mes,
            marketing,
            clientes
        ]],
        dtype=np.float32
    )

    diferenca = X_max - X_min

    diferenca[diferenca == 0] = 1

    entrada_normalizada = (
        entrada - X_min
    ) / diferenca

    previsao_normalizada = (
        modelo.predict(
            entrada_normalizada,
            verbose=0
        )[0][0]
    )

    previsao = (
        previsao_normalizada
        * (y_max - y_min)
        + y_min
    )

    return float(previsao)


# ============================================================
# APLICAÇÃO
# ============================================================

def main():

    st.title(
        "📈 Análise e Previsão de Vendas"
    )

    st.write(
        "Aplicação desenvolvida com "
        "**Python, Pandas, NumPy, TensorFlow e Streamlit**."
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    df = criar_dataframe()

    if df is None:
        return

    st.divider()

    # --------------------------------------------------------
    # ABAS
    # --------------------------------------------------------

    aba_dataset, aba_analise, aba_modelo = st.tabs(
        [
            "📋 Dataset",
            "📊 Análise",
            "🤖 Previsão"
        ]
    )

    # --------------------------------------------------------
    # ABA DATASET
    # --------------------------------------------------------

    with aba_dataset:

        st.subheader(
            "Dataset de vendas"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.write(
            f"Total de registros: **{len(df)}**"
        )

    # --------------------------------------------------------
    # ABA ANÁLISE
    # --------------------------------------------------------

    with aba_analise:

        analisar_dados(df)

    # --------------------------------------------------------
    # ABA PREVISÃO
    # --------------------------------------------------------

    with aba_modelo:

        st.subheader(
            "🤖 Modelo de previsão"
        )

        st.write(
            "O modelo utiliza TensorFlow para "
            "estimar o valor das vendas."
        )

        if st.button(
            "🚀 Treinar modelo",
            type="primary"
        ):

            with st.spinner(
                "Treinando o modelo..."
            ):

                (
                    modelo,
                    X_min,
                    X_max,
                    y_min,
                    y_max
                ) = treinar_modelo(df)

                st.session_state.modelo = modelo
                st.session_state.X_min = X_min
                st.session_state.X_max = X_max
                st.session_state.y_min = y_min
                st.session_state.y_max = y_max

            st.success(
                "Modelo treinado com sucesso! ✅"
            )

        st.divider()

        st.subheader(
            "🔮 Fazer previsão"
        )

        mes = st.number_input(
            "Mês",
            min_value=1,
            max_value=12,
            value=6,
            step=1
        )

        marketing = st.number_input(
            "Investimento em marketing (R$)",
            min_value=0.0,
            value=2500.0,
            step=100.0
        )

        clientes = st.number_input(
            "Número de clientes",
            min_value=0,
            value=200,
            step=10
        )

        if st.button(
            "📈 Prever vendas"
        ):

            if "modelo" not in st.session_state:

                st.warning(
                    "Primeiro treine o modelo."
                )

            else:

                try:

                    previsao = realizar_previsao(
                        st.session_state.modelo,
                        mes,
                        marketing,
                        clientes,
                        st.session_state.X_min,
                        st.session_state.X_max,
                        st.session_state.y_min,
                        st.session_state.y_max
                    )

                    st.success(
                        "Previsão realizada com sucesso!"
                    )

                    st.metric(
                        "💰 Vendas previstas",
                        f"R$ {previsao:,.2f}"
                    )

                except Exception as erro:

                    st.error(
                        f"Erro ao realizar previsão: {erro}"
                    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()


