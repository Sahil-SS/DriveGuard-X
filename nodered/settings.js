/** DriveGuard-X Node-RED runtime settings. */
module.exports = {
  uiPort: Number(process.env.PORT || 1880),
  httpNodeRoot: "/",
  httpAdminRoot: "/",
  flowFile: "flows.json",
  flowFilePretty: true,
  functionExternalModules: false,
  editorTheme: {
    projects: { enabled: false },
  },
};
