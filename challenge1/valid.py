import kedro_light as kl

from transformations import (
    join,
    prepare,
    encode,
    calculate_geometric_features,
    split,
    train,
    validate,
    score,
    plot,
    benchmark,
    skill,
    submit,
)


io = kl.io(conf_paths="conf", catalog="catalog.yml")
dag = [
    kl.node(func=join, inputs=["raw_train_x", "raw_train_y"], outputs="raw_train"),
    kl.node(func=prepare, inputs="raw_train", outputs="prep_train"),
    kl.node(func=encode, inputs="prep_train", outputs="enc_train"),
    kl.node(func=calculate_geometric_features, inputs="enc_train", outputs="train"),
    kl.node(func=train, inputs="train", outputs="regr"),
    kl.node(func=prepare, inputs="raw_valid_x", outputs="prep_valid"),
    kl.node(func=encode, inputs="prep_valid", outputs="enc_valid"),
    kl.node(func=calculate_geometric_features, inputs="enc_valid", outputs="valid"),
    kl.node(func=validate, inputs=["regr", "valid"], outputs="outputs_model"),
    kl.node(func=submit, inputs=["template_phase1", "outputs_model"], outputs="outputs_valid"),
]
kl.run(dag, io)
