import './App.css';
import { inference } from './inference.js';
import React from 'react';
import {modelDownloadInProgress} from './inference.js';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import LinearProgress from '@mui/material/LinearProgress';

class TextInputArea extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      text: 'Enter text to responds. Game was made to be played in English.',
      disabled: true,
      downloading:modelDownloadInProgress(),
      response: "Connor walks up to me and asks a question.\n Connor: Yo you up to talk right now?",
      possible_sequence_ids: ["sequence1", "sequence2", "sequence3"],
      current_sequence_name: "Game Start"
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

  getNewSequence(embedding) {
    fetch(`http://localhost:5000/sequence/${JSON.stringify(embedding)}`, {
      method: "GET",
      mode: "cors"
    })
      .then(async response => {
        const data = await response.json()
        for (let index = 0; index < data.ids[0].length; index++) {
          if (this.state.possible_sequence_ids.includes(data.ids[0][index])) {
            console.log(data.metadatas[0][index].reachableSequences)
            const possibleSequenceString = JSON.parse(data.metadatas[0][index].reachableSequences)
            this.setState({
              response: data.metadatas[0][index].reaction,
              current_sequence_name: data.documents[0][index],
              possible_sequence_ids: possibleSequenceString.sequences
            })
            return
          }
        }
      })
  }

  handleSubmitText(event) {  
    // Ensure that model will only receive valid text
    var regExp = /[a-zA-Z]/g;                
    if(regExp.test(this.state.value)){
      inference(this.state.value).then(embedding => {
        this.getNewSequence(embedding)
      })
    }
  }

  ping() {
    fetch("http://localhost:5000/ping")
  }

  render() {
    return (
      <div className="App">
      <header className="App-header">   
      <em>Cool Over-protective Dad</em>
      <div><font size="3">A game about a pretty cool dad, who happens to be a little over-protective</font></div>
      <div><font size="3">Writer: Connor Killingbeck</font></div>
      <div><font size="3">Programmer: Khoa Nguyen</font></div>

      {this.state.response}
      
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
      <Button onClick={this.ping}>Ping</Button>

      <div><font size="3">GitHub Repo: <a href="https://github.com/lifelike-toolkit/lifelike">browser-ml-inference</a></font></div>
      <div><font size="3">Some code repurposed from: <a href="https://github.com/jobergum/browser-ml-inference">browser-ml-inference</a></font></div>
      <div><font size="3">Model was trained on the <a href="https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html">GoEmotions</a> dataset.</font></div>
      </header>
    </div>   
    );
  }
}
export default TextInputArea;
