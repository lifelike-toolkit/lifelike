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
      response: "Inner Dialogue: The air is cool and the streets are lively as I approach the building. It has been many years since I found myself in this part of town. Those were definitely different times, being young and bold, sometimes I wish I could just go back to that part in my life again\u2026although tonight, searching for that link to my younger self is not my intention. The line is long as I approach the city's most famous club, The Conman\u2122 . The drinks, the music, the people, a building filled with dancing floors, lounges, weight rooms, and even an aquarium to boot, the ultimate clubbing experience. Me and the Boys used to come here every weekend, and seemingly so did everyone from all over the galaxy, The Conman\u2122  is just that kind of place. Honestly, it\u2019s not surprising that my daughter has decided to come here. I would compliment her good taste if it wasn't past curfew and she hadn\u2019t snuck out of the house to be here. How could she do such a thing? I\u2019ll just have to ask her when I find her I suppose. I take a deep breath to compose myself as I inch forwards in the line. Of course by the time I arrive it is peak hours, on a Saturday no less. Everyone is going to be here for what is bound to be an insane and unreal experience, and that means that the line is going to be equally brutal. Finally, after what seems like hours, I make it to the front of the line. Looks like you don't have to listen to any more introductory exposition! 3 paragraphs honestly isn\u2019t even that bad, sometimes I impress myself. The bouncer summons me forwards and I oblige, walking towards him to the very front of the building. The sheer size of it is incredible, it being almost the size of a skyscraper, if only the AI generated art for this part of the game allowed you to see it! As I approach, the bouncer gestures his hand for me to stop in place, and finally, character dialogue begins! \\nBouncer: Good evening sir, thank you for choosing The Conman\u2122 as you place for partying! \\nInner Dialogue: The Bouncer looks me up and down briefly \\nBouncer: GEEZ, you are impressive my man! That style and that body frame make you quite the protagonist! \\nC.O.D: Thanks, I got very lucky with the random art generation. \\nBouncer: Yeah I\u2019ll say! You definitely look of age, what are your plans for tonight sir?".split('\\n'),
      possible_sequence_ids: ["event1", "event2", "event3"],
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
    alert("The toolkit this game runs on allows dev to use any model. Unfortunately, for the purpose of the demo, our model cannot interpret context clues. Try to give the games dialogues that can be taken out of context and its sentiment can still make sense. Also, long paragraph will overload the model. Have fun!")
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
              response: data.metadatas[0][index].reaction.split("\\n"),
              current_sequence_name: data.documents[0][index],
              possible_sequence_ids: possibleSequenceString
            })
            return
          }
        }
      })
  }

  handleSubmitText(event) {  
    if (this.state.possible_sequence_ids.length === 0) {
      alert("Congrats! You've reach an ending: " + this.state.current_sequence_name + ". Refresh page to try again and find all hidden ending")
    }
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
      <em>Cool Overprotective Dad: Chapter 1</em>
      <div><font size="3">A game about a pretty cool dad, who happens to be a little overprotective.</font></div>
      <div><font size="3">Follow <a href="https://twitter.com/mustafa_tariqk">Mustafa Tariq on Twitter</a> to see when the rest of the game is available</font></div>
      <div><font size="3">Made with lifelike: <a href="https://github.com/lifelike-toolkit/lifelike">lifelike-toolkit</a></font></div>
      <div><font size="3">Director & Programmer: Khoa Nguyen</font></div>
      <div><font size="3">Writer & Creative Director: Connor Killingbeck</font></div>

      {this.state.response.map((i,key) => {
          let diag_arr = i.split(':')
          return <div align='justify' padding='3px' key={key}><b>{diag_arr[0] + ':'}</b>{diag_arr[1]}</div>;
      })}
      
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

      <div><font size="3">GitHub Repo: <a href="https://github.com/lifelike-toolkit/lifelike">lifelike-toolkit</a></font></div>
      <div><font size="3">Some code repurposed from: <a href="https://github.com/jobergum/browser-ml-inference">browser-ml-inference</a></font></div>
      <div><font size="3">Model was trained on the <a href="https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html">GoEmotions</a> dataset.</font></div>
      </header>
    </div>   
    );
  }
}
export default TextInputArea;
