// Use CommonJS for Jest config, and set up Babel for transforming JS files
module.exports = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  verbose: true,
};
