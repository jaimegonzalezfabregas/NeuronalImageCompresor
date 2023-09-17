
module.exports = {
    src_img: "../workspace/sea.jpeg",
    evolution_path_hypersampled: "../workspace/evolution_hypersampled/",
    evolution_path: "../workspace/evolution/",
    model_path: "../workspace/compresion_models/",
    extrapolation_path: "../workspace/extrapolations/",
    get_virgin_model: (tf) => {
        const { point_to_input } = require("./utils");

        let layers = [
            tf.layers.dense({ inputShape: [point_to_input(0, 0).length], units: 50, activation: 'selu' }),

        ];

        for (let i = 0; i < 10; i++) {

            layers.push(
                tf.layers.dense({ units: 100, activation: 'selu' }),
                tf.layers.dense({ units: 100, activation: 'selu' }),
                tf.layers.dense({ units: 3, activation: 'sigmoid' }),
            )
        }

        layers.push()

        const model = tf.sequential({
            layers,
        });

        return model;
    }
}