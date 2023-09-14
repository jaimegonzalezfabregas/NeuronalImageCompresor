const { rgb_to_file, point_to_input } = require("./utils");

const decompress = async (model, dst_path, width, height, tf) => {

    console.log(dst_path, width, height)

    let coord_pairs = []

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            coord_pairs.push(point_to_input(x / width, y / height))
        }
    }

    const test_ins = tf.tensor(coord_pairs);

    console.log("model predict")

    const net_out = await model.predict(test_ins);
    console.log("reshaping")

    const result = await net_out.array()

    let rgb_image = [];
    for (const pixel of result) {
        // console.log(pixel)
        rgb_image.push(...pixel)
    }
    console.log("writing to disk")

    rgb_to_file(rgb_image, width, height, dst_path);

}


if (!module.parent) {
    const tf = require("@tensorflow/tfjs-node");

    (async () => {
        const model = await tf.loadLayersModel('file://compresion_model/model.json');
        decompress(model, "dst.jpeg", 666, 1000, tf)
    })()
}

module.exports = decompress
