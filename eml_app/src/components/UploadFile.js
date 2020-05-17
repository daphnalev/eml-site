import React, { Component } from "react";
import { Button } from "react-bootstrap";

class UploadFile extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: []
    };
  }

  componentDidMount() {
    
  }

  render() {
    return (
      <div>
        <h4>Please upload your file</h4>
        <Button onClick={this.props.fileUploaded}>Upload</Button>
      </div>
    );
  }
}

export default UploadFile;