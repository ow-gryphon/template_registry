const fs = require('fs-extra');
const path = require('path');
const _ = require('lodash');

module.exports = {
  description: 'adds a new set of .py files to create an example clustering problem',
  templatesFolder: path.resolve(__dirname, './template'),
  arguments: [
    { name: 'name', required: true },
  ],
  parseArguments: (name) => {
    const commandName = _.snakeCase(name);
    // Execute the command
    const commandArgs = {
      commandName,
      fileName: commandName,
      folder: commandName,
    };

    // returns an array of the "commandArgs" expected by the CLI commands/generate.js
    return [commandArgs];
  },
  examples: [
    { command: 'lk generate ml-clustering mymlclustering', description: 'add new ml-clustering to current project' },
  ],
  onGenerationSuccess: () => {
    fs.readFile('readme_mlclustering.md', 'utf8', (err, data) => {
      if (err === null) console.log(data);
    });
  }
}