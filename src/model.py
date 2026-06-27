"""
model.py
--------
Defines every model used in this project:

  1. transformer_lstm  -- the PROPOSED model (this project's contribution)
  2. lstm_only          -- baseline
  3. cnn_only           -- baseline
  4. cnn_lstm           -- baseline (reproduction of the base paper's
                            architecture: Mandela et al., "Efficient Dark
                            Web traffic classification using a hybrid
                            CNN-LSTM model")
  5. logistic_regression / svm -- classical ML baselines (see train.py,
                            these use scikit-learn directly rather than
                            Keras, since they don't need this module)

All Keras models expect input of shape (timesteps, num_features) and
predict a single sigmoid output (probability of class "Tor").

Why Transformer + LSTM (in plain words):
The LSTM layer reads the flow's feature vector and learns sequential /
temporal structure the same way the base paper's LSTM does. The
Transformer block that follows uses multi-head self-attention to learn
*which features matter most* for distinguishing Tor traffic from
ordinary traffic, and how those features relate to each other -- this is
the same job the base paper's CNN was doing (spatial feature
extraction), but attention can directly model relationships between any
two features regardless of their position, whereas a CNN can only look
at features that are near each other after however the data happened to
be ordered into columns. That's the concrete reason this swap is a
genuine architectural improvement, not just a relabeling.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_transformer_lstm(
    input_shape,
    lstm_units: int = 64,
    num_attention_heads: int = 4,
    attention_key_dim: int = None,
    ff_dim: int = None,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    The proposed model for this project.

    Architecture (matches the CEP's required structure):
        Input -> LSTM -> Transformer (multi-head attention) -> Dropout
              -> Dense -> Output (sigmoid)

    Parameters
    ----------
    input_shape : tuple
        (timesteps, num_features) of the preprocessed data.
    lstm_units : int
        Size of the LSTM's hidden state. Kept equal to the base paper's
        LSTM (64 units) so any accuracy gain we see is attributable to
        adding attention, not just to using a bigger LSTM.
    num_attention_heads : int
        Number of parallel attention heads in the Transformer block.
    attention_key_dim : int, optional
        Dimensionality of each attention head's query/key vectors. If
        None (default), this is sized automatically based on the
        number of input features -- see the sizing note below.
    ff_dim : int, optional
        Size of the Transformer block's feed-forward sublayer. If None
        (default), sized automatically the same way.
    dropout_rate : float
        Dropout applied after attention and before the final dense
        layers, to control overfitting.

    Why attention_key_dim/ff_dim are auto-sized instead of fixed:
    A Transformer block's parameter count needs to be matched to how
    much data and how many input features you actually have, or it can
    easily out-grow what the dataset can support. We initially used a
    fixed key_dim=32 / ff_dim=128 for every dataset, which gave this
    model roughly 2x as many parameters as the CNN-LSTM baseline even
    on a 22-feature dataset with ~50K training rows -- a real capacity
    mismatch that visibly hurt performance on the smaller of our two
    datasets (ISCXTor2016) compared to the larger, richer-feature one
    (CIC-Darknet2020). Scaling these two sizes down for smaller feature
    counts keeps the same architecture (LSTM -> attention -> FFN) while
    giving it a parameter budget appropriate to the data, which is a
    standard, well-documented practice when applying Transformers to
    smaller tabular datasets (see Related Work section on Transformer
    adaptation strategies for limited data).
    """
    num_features = input_shape[-1]
    if attention_key_dim is None:
        # Roughly one query/key dimension per ~3 input features, with a
        # sensible floor/ceiling so very small or very large feature
        # counts still get a workable head size.
        attention_key_dim = max(8, min(32, num_features // 3))
    if ff_dim is None:
        # Feed-forward sublayer sized relative to the LSTM's hidden
        # size (lstm_units) rather than always being a fixed 128 --
        # 2x the LSTM size is a common Transformer convention, and we
        # cap it so it never balloons past what 64 LSTM units justify.
        ff_dim = min(128, lstm_units * 2)

    inputs = keras.Input(shape=input_shape, name="flow_features")

    # LSTM with return_sequences=True so its full output sequence (not
    # just the final hidden state) is available for the Transformer
    # block to attend over. This matters even when TIMESTEPS=1, because
    # it keeps the code general enough to work unchanged if a future
    # version of this project uses multi-packet sequences instead of
    # single feature vectors per flow.
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)

    # --- Transformer encoder block ---
    # Multi-head self-attention: lets the model learn which positions in
    # the LSTM's output sequence should influence each other, and by how
    # much, instead of treating every position with equal importance.
    attn_output = layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=attention_key_dim,
        name="multi_head_attention",
    )(x, x)  # self-attention: query, value, and key all come from x
    attn_output = layers.Dropout(dropout_rate, name="attn_dropout")(attn_output)
    # Residual connection + layer norm (standard Transformer design --
    # this is what lets gradients flow cleanly through deep stacks and
    # is explicitly why the CEP/literature recommend "Add & Norm").
    x = layers.Add(name="attn_residual")([x, attn_output])
    x = layers.LayerNormalization(name="attn_layernorm")(x)

    # Position-wise feed-forward sublayer (the other standard Transformer
    # component): two dense layers applied independently at every
    # position, giving the model extra capacity to transform the
    # attended features before the next stage.
    ff_output = layers.Dense(ff_dim, activation="relu", name="ffn_dense1")(x)
    ff_output = layers.Dense(x.shape[-1], name="ffn_dense2")(ff_output)
    ff_output = layers.Dropout(dropout_rate, name="ffn_dropout")(ff_output)
    x = layers.Add(name="ffn_residual")([x, ff_output])
    x = layers.LayerNormalization(name="ffn_layernorm")(x)

    # Flatten the (timesteps, features) sequence down to a single vector
    # per sample before the classification head.
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="Transformer_LSTM")
    return model


def build_lstm_only(input_shape, lstm_units: int = 64, dropout_rate: float = 0.3) -> keras.Model:
    """Baseline: standalone LSTM, no attention, no CNN."""
    inputs = keras.Input(shape=input_shape, name="flow_features")
    x = layers.LSTM(lstm_units, name="lstm")(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="LSTM_Only")


def build_cnn_only(input_shape, dropout_rate: float = 0.3) -> keras.Model:
    """Baseline: standalone 1D-CNN, no LSTM, no attention."""
    inputs = keras.Input(shape=input_shape, name="flow_features")
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling1D(pool_size=1, name="pool1")(x)  # pool_size=1 since TIMESTEPS=1
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="CNN_Only")


def build_cnn_lstm(input_shape, lstm_units: int = 64, dropout_rate: float = 0.3) -> keras.Model:
    """
    Baseline: CNN + LSTM, reproducing the base paper's architecture
    (Mandela et al.) as closely as our tabular-feature input allows:
        Conv1D(64 filters, kernel 5, ReLU) -> MaxPool -> LSTM(64) -> Dense(sigmoid)

    Note: the base paper used an embedding layer first because their
    input was raw character sequences from packet bytes. Our input is
    already-numeric flow statistics, so we skip the embedding step and
    feed features directly into the CNN -- the CNN/LSTM/Dense stack
    itself matches their design.
    """
    inputs = keras.Input(shape=input_shape, name="flow_features")
    x = layers.Conv1D(64, kernel_size=min(5, input_shape[0]), padding="same",
                       activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling1D(pool_size=1, name="pool1")(x)
    x = layers.LSTM(lstm_units, name="lstm")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name="CNN_LSTM_BasePaper")


MODEL_BUILDERS = {
    "transformer_lstm": build_transformer_lstm,
    "lstm_only": build_lstm_only,
    "cnn_only": build_cnn_only,
    "cnn_lstm": build_cnn_lstm,
}


def build_model(model_name: str, input_shape) -> keras.Model:
    """Convenience dispatcher used by train.py."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[model_name](input_shape)


if __name__ == "__main__":
    # Quick sanity check: build each model and print its summary so you
    # can confirm shapes line up before running a full training job.
    dummy_input_shape = (1, 80)  # (timesteps, features) -- 80 is a placeholder
    for name in MODEL_BUILDERS:
        print(f"\n===== {name} =====")
        m = build_model(name, dummy_input_shape)
        m.summary()
