const { model_path } = require("./config");
const { rgb_to_file, point_to_input } = require("./utils");

const decompress = async (model, dst_path, input_width, input_height, tf, x0, y0, xf, yf) => {

    let width = Math.floor(input_width)
    let height = Math.floor(input_height)

    console.log(dst_path, width, height)

    let coord_pairs = []

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            coord_pairs.push(point_to_input(x / width, y / height, x0, y0, xf, yf))
        }
    }

    const test_ins = tf.tensor(coord_pairs);

    console.log("model predict", coord_pairs.length)

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

    test_ins.dispose()
    net_out.dispose()
}

module.exports = decompress
