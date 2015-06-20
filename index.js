var express = require('express');
var cors = require('express-cors');
//var bodyParser = require('body-parser');
var app = express();

// allow CORS
app.use(cors({
    allowedOrigins: [
        '*:*'
    ]
}));


var server = app.listen(3000, 'localhost', function () {

  var host = server.address().address;
  var port = server.address().port;

  console.log('Example app listening at http://%s:%s', host, port);

});