export default function txtLoaderPlugin() {
  return {
    name: 'txt-loader-plugin',
    /**
     * Ensure Webpack can handle plain `.txt` files like `llms.txt` without
     * trying to parse them as JavaScript.
     */
    configureWebpack() {
      return {
        module: {
          rules: [
            {
              test: /\.txt$/i,
              type: 'asset/source',
            },
          ],
        },
      };
    },
  };
}
