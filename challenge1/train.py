import kedro_light as kl

from transformations import (
    upsample,
    join_x,
    join_y,
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
    feature_importance,
)


io = kl.io(conf_paths="conf", catalog="catalog.yml")
dag = [
    kl.node(func=upsample, inputs="raw_weather", outputs="weather"),
    kl.node(func=join_x, inputs=["raw_train_x", "weather"], outputs="train_x"),
    kl.node(func=join_y, inputs=["train_x", "raw_train_y"], outputs="raw"),
    kl.node(func=prepare, inputs="raw", outputs="prep"),
    kl.node(func=encode, inputs="prep", outputs="enc"),
    kl.node(func=split, inputs="enc", outputs=["enc_train", "enc_valid"]),
    kl.node(func=calculate_geometric_features, inputs="enc_train", outputs="train"),
    kl.node(func=calculate_geometric_features, inputs="enc_valid", outputs="valid"),
    kl.node(func=train, inputs="train", outputs="regr"),
    kl.node(func=validate, inputs=["regr", "valid"], outputs="outputs_model"),
    kl.node(func=score, inputs="outputs_model", outputs="score_model"),
    kl.node(func=plot, inputs="outputs_model", outputs="figure"),
    kl.node(func=benchmark, inputs="valid", outputs="outputs_benchmark"),
    kl.node(func=score, inputs="outputs_benchmark", outputs="score_benchmark"),
    kl.node(func=skill, inputs=["score_model", "score_benchmark"], outputs="score"),
    kl.node(func=feature_importance, inputs=["regr", "train"], outputs="tmp"),
]
kl.run(dag, io)
