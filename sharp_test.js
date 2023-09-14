const { rbg_from_file, rgb_to_file } = require("./utils");

rbg_from_file("src.jpeg").then(({ data, info }) => {
    rgb_to_file(data, info.width, info.height, "test.jpeg")
})