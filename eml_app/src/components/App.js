import React, { Component } from "react";
import { render } from "react-dom";
import UploadFile from "./UploadFile"
import FileConfiguration from "./FileConfigoration"

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      file: false
    };

    this.fileUploaded = this.fileUploaded.bind(this)
  }

  fileUploaded(file) {
    console.log("file uploaded");
    this.setState({ file: true });
  }

  render() {
    return (
      <div>
        <UploadFile fileUploaded={this.fileUploaded} />
        {this.state.file ? <FileConfiguration /> : null}
      </div>
    );
  }
}

export default App;
render(<App />, document.getElementById("app"));