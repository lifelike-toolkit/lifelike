import './App.css';

import React from 'react';
import {inference} from './inference.js';
import {modelDownloadInProgress} from './inference.js';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import LinearProgress from '@mui/material/LinearProgress';

class TextInputArea extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      text: 'Enter text to classify emotion, model trained on English text.',
      disabled: true,
      downloading:modelDownloadInProgress()
    };
    this.handleChangeText = this.handleChangeText.bind(this);
    this.handleSubmitText = this.handleSubmitText.bind(this);
  }

  componentDidMount() {
    this.timerID = setInterval(
      () => this.checkModelStatus(),
      1000
    );
  }

  checkModelStatus() {
    this.setState({
      downloading: modelDownloadInProgress(),
    });
    if (!this.state.downloading) {
      this.timerID = setInterval(
        () => this.checkModelStatus,
        5000000
      );
    }
  }

  handleChangeText(event) {
    var regExp = /[a-zA-Z]/g;                
    if(regExp.test(event.target.value)){
      this.setState({
        value: event.target.value,
        disabled: false
      });
    } else {
      this.setState({
        value: event.target.value,
        disabled: true
      });
    }
  }

  handleSubmitText(event) {  
    // Ensure that model will only receive valid text
    var regExp = /[a-zA-Z]/g;                
    if(regExp.test(this.state.value)){
      inference(this.state.value).then( result => {
        this.setState({
          data: result,
        });
      });
    }
  }

  render() {
    return (
      <div className="App">
      <header className="App-header">   
      <em>In-Browser Transformer Inference</em>
      {this.state.data ? this.state.data.map(data => data + "\n") : null}
      
      {this.state.downloading && 
        <div><font size="2">Downloading model from CDN to browser..</font>
        <Box sx={{ width: '400px' }}>
        <LinearProgress />
        </Box> 
        <p></p>
        </div>
      }
      <textarea rows="8" cols="24" className="App-textarea" name="message" 
       placeholder={this.state.text} autoFocus onChange={this.handleChangeText}>
      </textarea>
      <Button variant='contained' disabled={this.state.disabled} onClick={this.handleSubmitText}>Say it</Button>

      <div><font size="3">GitHub Repo: <a href="https://github.com/lifelike-toolkit/lifelike">browser-ml-inference</a></font></div>
      <div><font size="3">Some code repurposed from: <a href="https://github.com/jobergum/browser-ml-inference">browser-ml-inference</a></font></div>
      <div><font size="3">Model was trained on the <a href="https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html">GoEmotions</a> dataset.</font></div>
      </header>
    </div>   
    );
  }
}
export default TextInputArea;
