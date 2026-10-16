const path = require('path');

// The face recognizer is vendored from margelo/face-recognition-demo (MIT) and linked as a workspace.
module.exports = {
  dependencies: {
    'react-native-vision-camera-face-recognizer': {
      root: path.join(__dirname, 'modules', 'react-native-face-recognition'),
    },
  },
};
