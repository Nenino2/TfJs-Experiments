import * as DICTIONARY from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/dictionary.js';
const POST_COMMENT_BTN = document.getElementById('post');
const COMMENT_TEXT = document.getElementById('comment');
const COMMENTS_LIST = document.getElementById('commentsList');
const PROCESSING_CLASS = 'processing';

const MODEL_JSON_URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/model.json';
const SPAM_THRESHOLD = 0.75;
const ENCODING_LENGTH = 20;
var model = undefined;



function handleCommentPost() {
    // Only continue if you are not already processing the comment.
    if (! POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
        POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
        COMMENT_TEXT.classList.add(PROCESSING_CLASS);
        let currentComment = COMMENT_TEXT.innerText;
        let lowercaseSentenceArray = currentComment.toLowerCase().replace(/[^\w\s]/g, '').split(' ');
        let li = document.createElement('li');
        let p = document.createElement('p');
        p.innerText = COMMENT_TEXT.innerText;
        let spanName = document.createElement('span');
        spanName.setAttribute('class', 'username');
        spanName.innerText = "Gino perugino";
        let spanDate = document.createElement('span');
        spanDate.setAttribute('class', 'timestamp');
        let curDate = new Date();
        spanDate.innerText = curDate.toLocaleString();
        li.appendChild(spanName);
        li.appendChild(spanDate);
        li.appendChild(p);
        COMMENTS_LIST.prepend(li); 
        COMMENT_TEXT.innerText = '';   

        loadAndPredict(tokenize(lowercaseSentenceArray), li).then(function() {
        
          POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
        
          COMMENT_TEXT.classList.remove(PROCESSING_CLASS);
        
        });
  
    }
}
  
POST_COMMENT_BTN.addEventListener('click', handleCommentPost);

async function loadAndPredict(inputTensor, domComment) {
    if (!model) {
        model = await tf.loadLayersModel(MODEL_JSON_URL)
    }

    const results = model.predict(inputTensor)

    results.print()

    let dataArray = results.dataSync()

    if (dataArray[1] > SPAM_THRESHOLD) {
        domComment.classList.add("spam")
    } else {
        socket.emit('comment', {

            username: "Gino perugino",
      
            timestamp: domComment?.querySelectorAll('span')[1].innerText,
      
            comment: domComment?.querySelectorAll('p')[0].innerText
      
        });
    }

}

function tokenize(wordArray) {
    let returnArray = [DICTIONARY.START]

    for (let word of wordArray) {
        let encoding = DICTIONARY.LOOKUP[word]
        returnArray.push(encoding ? encoding : DICTIONARY.UNKNOWN)
    }

    while (returnArray.length < ENCODING_LENGTH) {
        returnArray.push(DICTIONARY.PAD)
    }

    console.log(returnArray)

    return tf.tensor2d([returnArray])
}


let socket = io.connect();


function handleRemoteComments(data) {

  let li = document.createElement('li');

  let p = document.createElement('p');

  p.innerText = data.comment;


  let spanName = document.createElement('span');

  spanName.setAttribute('class', 'username');

  spanName.innerText = data.username;


  let spanDate = document.createElement('span');

  spanDate.setAttribute('class', 'timestamp');

  spanDate.innerText = data.timestamp;


  li.appendChild(spanName);

  li.appendChild(spanDate);

  li.appendChild(p);

  

  COMMENTS_LIST.prepend(li);

}


socket.on('remoteComment', handleRemoteComments);