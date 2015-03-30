% train.m
% Naive Bayes text classifier

clear all; close all; clc

% store the number of training examples
numTrainDocs =700;
%70;%140;350;700;

% store the dictionary size
numTokens = 19723;
%11416%12838;%16701;22745;
%19723;20520;x

numTestDocs = 52;
%52;130;260

% read the features matrix
M = dlmread('output_test_26/train-features.txt', ',');
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens);
train_matrix = full(spmatrix);

% train_matrix now contains information about the words within the emails
% the i-th row of train_matrix represents the i-th training email
% for a particular email, the entry in the j-th column tells
% you how many times the j-th dictionary word appears in that email

% read the training labels
train_labels = dlmread('output_test_26/train-labels.txt');
% the i-th entry of train_labels now indicates whether document i is spam


% Find the indices for the spam and nonspam labels
spam_indices = find(train_labels);
nonspam_indices = find(train_labels == 0);

% Sum the number of words in each email by summing along each row of
% train_matrix
email_lengths = sum(train_matrix, 2);
% Now find the total word counts of all the spam emails and nonspam emails
spam_wc = sum(email_lengths(spam_indices));
nonspam_wc = sum(email_lengths(nonspam_indices));

%{
% It is not right but I leave it here
features_non_spam = zeros(100, numTokens);
% fucken matlab
% 100 * (M[i,j] / email.length[i])     j is email num

for j = 1:numTrainDocs/2
    for i = j:numTokens
        features_non_spam(i) = train_matrix(j,i)/email_lengths(j);
    end
end

features_spam = zeros(100, numTokens);
for j = numTrainDocs/2:numTrainDocs
    for i = j:numTokens
        features_spam(i) = train_matrix(j,i)/email_lengths(j);
    end
end
%}

SVMStruct = svmtrain(train_matrix,train_labels,'showPlot',true);


N = dlmread('output_test_26/test-features.txt', ',');
spmatrix = sparse(N(:,1), N(:,2), N(:,3), numTestDocs, numTokens);
test_matrix = full(spmatrix);

output = svmclassify(SVMStruct,test_matrix);



% Read the correct labels of the test set
test_labels = dlmread('output_test_26/test-labels.txt');

% Compute the error on the test set
% A document is misclassified if it's predicted label is different from
% the actual label, so count the number of 1's from an exclusive "or"
numdocs_wrong = sum(xor(output, test_labels))

%Print out error statistics on the test set
fraction_wrong = numdocs_wrong/numTestDocs
