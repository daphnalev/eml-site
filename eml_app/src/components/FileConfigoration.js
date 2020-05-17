import React, { Component } from "react";
import { render } from "react-dom";

class FileConfiguration extends Component {
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
      <p>Change Configuration</p>
    );
  }
}

export default FileConfiguration;
