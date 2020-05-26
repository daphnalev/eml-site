import React from "react";
import { Button } from "react-bootstrap";
import ReactFileReader from 'react-file-reader';
import CSVReader from 'react-csv-reader'

const papaparseOptions = {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
  transformHeader: header =>
    header
      .toLowerCase()
      .replace(/\W/g, '_')
}

class UploadFile extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      file: null,
      uploaded: false
    };

    this.uploaded = this.uploaded.bind(this);
  }

  uploaded(newFile) {
    this.setState({
      file: newFile,
      uploaded: true
    });

    console.log(newFile.base64);
    this.props.fileUploaded(newFile);
  }

  onFileLoaded(data, fileInfo) {
    console.log(data);
    console.log(fileInfo);

    //<CSVReader
    //cssClass = "csv-reader-input"
    //label = "Select CSV with secret Death Star statistics"
    //onFileLoaded = { this.handleForce }
    //onError = { this.handleDarkSideForce }
    //parserOptions = { papaparseOptions }
    //inputId = "ObiWan"
    //inputStyle = {{ color: 'red' }}
    //   />
  }

  render() {
    return (
      <div>
        {!this.state.uploaded ?
          <center>
            <h4>Please upload your file</h4>
            <ReactFileReader fileTypes={[".csv"]} base64={true} handleFiles={this.uploaded}>
              <button>Upload</button>
            </ReactFileReader>
          </center>
          :
          <center>
            <h4>Read</h4>
            <iframe src={this.state.file} frameBorder="0" height="400" width="50%" />
          </center>}
      </div>
    );
  }
}

export default UploadFile;