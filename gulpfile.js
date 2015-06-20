'use strict';

var path = require('path');
var gulp = require('gulp');
var less = require('gulp-less');
var clean = require('gulp-clean');
var minifycss    = require('gulp-minify-css');
var connect = require('gulp-connect');

//CONFIG PATHS
var config = {
	pages  : './pages',
	assets : './assets',
	build:'./dist'
};

//TASKS
gulp.task('less', function () {
	gulp.src(config.pages+'/less/pages.less')
		.pipe(less({
				paths: [config.pages+'/less/']
		}))
		.pipe(gulp.dest(config.pages+'/css/'))
		.pipe(connect.reload());
});
gulp.task('watch', function () {
	gulp.watch(config.pages+'/**/*.less', function(event) {
		gulp.run('less');
	});
});

gulp.task('build',['less','copy'],function() {
	gulp.run('css-min');

});

gulp.task('clean', function(){
	return gulp.src( config.build+'' , {read: false})
		.pipe(clean());
});

gulp.task('copy', ['clean'],function () {
	return gulp.src(['./**/*','!**/node_modules/**','!.gitgnore','!package.json','!Gruntfile.js','!gulpfile.js'])
	.pipe(gulp.dest(config.build+''));
});

gulp.task('css-min', function(){
	return gulp.src( [config.build+'./assets/css/*.css' , config.build+'./pages/css/*.css'])
		.pipe(minifycss());
});

gulp.task('serve', function() {
  connect.server({
    port: 8888,
		livereload: true
  });
});

gulp.task('default', ['build', 'serve'], function() {
 console.log( "\nServing the site on http://localhost:8888 \n" );
});