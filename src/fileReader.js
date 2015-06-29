

/*
function updateImage(dataRow) {

  var imgSVG = d3.select("#imageGenerator svg g");

  //data join
  var rect = imgSVG.selectAll('rect')
                   .data(dataRow.values);

  //note: assume square, so image is dimension*dimension
  var dimension = math.sqrt(dataRow.values.length);
  var pixelDimensions = 10;

  //update old elements as needed [only if enter().append("rect") is removed]

  function updatePixels(elem) {
    elem.attr({
    //rect.enter().append("rect")
        //.attr({
          width: pixelDimensions,
          height: pixelDimensions,
          x: function(d, i) {
            return pixelDimensions * parseInt(i % dimension);
          },
          y: function(d, i) {
            return pixelDimensions * parseInt(i / dimension);
          }
        }).style({
          fill: function (d,i) {
            return "rgb(" + d + "," + d + "," + d + ")";
          }
        });
  }

  updatePixels(rect);
  
  updatePixels(rect.enter().append("rect"));
    
  rect.exit().remove();

}



//d3.csv("./data/mnist_train_sample.csv", function(data) {
d3.text("./data/mnist_train_sample.csv", function(text) {
                      training_data = d3.csv.parseRows(text)
                                        .map(function(row) {
                                          return {
                                            result: +row.slice(0,1),
                                            values: row.slice(1).map(function(value) { return +value; })
                                          };
                                        });

                      var svg = d3.select("#imageGenerator")
                                  .append("svg")
                                  .attr({
                                    width: 280,
                                    height: 280
                                  })
                                  .append("g");

                      updateImage(training_data[0]);

                      //readyCheck("mnist_train_sample.csv");
                    });
*/

//var readyFiles = {'mnist_train_sample.csv': false, 'mnist_train_sample.csv': false};
//function readyCheck(filename) {
  //readyFiles[filename] = true;
//
  //if (readyFiles['mnist_train_sample']==true && readyFiles['mnist_train_sample']==true) {
    ////...begin drawing visualization...
//
  //}
//}