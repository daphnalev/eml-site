import React from "react";
import { Form, Button } from 'react-bootstrap';

class FileConfiguration extends React.Component {
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
      <Form>
        <Form.Group>
          <Form.Label>Choose Y column:</Form.Label>
          <Form.Control type="text" placeholder="Y column" />
          <Form.Text className="text-muted">
            The label of the sample
          </Form.Text>
        </Form.Group>

        <Form.Group>
          <Form.Label>Choose X columns:</Form.Label>
          <Form.Control type="text" placeholder="Y column" />
          <Form.Text className="text-muted">
            The attriuts of the sample
          </Form.Text>
        </Form.Group>

        <Form.Group>
          <Form.Label>Choose empty cell indiator:</Form.Label>
          <Form.Control type="text" placeholder="Empty cell" />
          <Form.Text className="text-muted">
            How do you mark that the data is missing?
          </Form.Text>
        </Form.Group>


        <Button variant="primary"> 1. Process Data </Button>
        <Button variant="primary" type="submit"> 2. Learn </Button>
      </Form>
    );
  }
}

export default FileConfiguration;
