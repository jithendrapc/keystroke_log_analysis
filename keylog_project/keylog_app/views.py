from django.contrib.auth import login
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import login_required, user_passes_test
from .forms import SignUpForm
from django.contrib import messages
import pickle
#from flask import Flask,request,render_template,jsonify
import numpy as np
import csv
import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

def is_admin(user):
    return user.is_authenticated and user.is_admin

def frontpage(request):
    return render(request, 'keylog_app/frontpage.html')

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)

        if form.is_valid():
            user = form.save()

            login(request, user)

            return redirect('frontpage')
    else:
        form = SignUpForm()
    
    return render(request, 'keylog_app/signup.html', {'form': form})


def user(request):
    return render(request,'keylog_app/user.html')

def admin(request):
    return render(request,'keylog_app/admin.html')

def index(request):
    return render(request,'keylog_app/index.html')

def home(request):
    return render(request,'keylog_app/home.html')



def predict_datapoint(request):
    if request.method == "GET":
        logging.info('Score is predicted.')
        return render(request,'home.html')
    else:
        data =  CustomData(
            jsonobject = request.POST.get('jsonobject')
        )
        prediction_df = data.get_data_as_data_frame()
        print(prediction_df)
        
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(prediction_df)
        context = {'prediction': prediction[0]}
        logging.info('score is predicted.')
        return render(request,'keylog_app/home.html',context)
    
    


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # Redirect to the home page or any desired URL
            request.session['username'] = username
            return redirect('home')
        else:
            # Handle invalid login credentials
            return HttpResponse('Invalid login credentials')

    return render(request, 'keylog_app/login.html')



def admin_login(request):
    return render(request, 'keylog_app/admin_login.html')

def admin_login_view(request):
    if request.method == 'POST':
        admin_id = request.POST.get('admin_id')
        password = request.POST.get('password')

        # Path to your admins.csv file
        csv_file_path = r'C:\Users\prath\Downloads\keystroke\keylog_project\keylog_app\admins.csv'

        # Check if the provided credentials match any entry in the CSV file
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row['admin_id'] == admin_id and row['password'] == password:
                    # Authentication successful
                    messages.success(request, 'Admin login successful.')
                    # Set the admin ID in the session for later use
                    request.session['admin_id'] = admin_id
                    return redirect('admin_dashboard')  # Redirect to admin dashboard or any other page
            
        # Authentication failed
        messages.error(request, 'Invalid admin credentials.')
    
    return render(request, 'keylog_app/admin_login.html')

def admin_dashboard(request):
    if request.method == 'POST':
        admin_id = request.session.get('admin_id')
        paragraph = request.POST.get('paragraph')
        mark = request.POST.get('mark')

        # Path to your data.csv file
        csv_file_path = 'questions.csv'

        # Write the input to the CSV file along with the admin ID
        with open(csv_file_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([admin_id, paragraph, mark])
            
        # Provide feedback to the admin
        messages.success(request, 'Paragraph and mark stored successfully.')
    
    return render(request, 'keylog_app/admin_dashboard.html')

def question(request):
    if request.method == 'POST':
        admin_id = request.session.get('admin_id')
        paragraph = request.POST.get('paragraph')
        mark = request.POST.get('mark')

        # Path to your data.csv file
        csv_file_path = 'questions.csv'

        # Write the input to the CSV file along with the admin ID
        with open(csv_file_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([admin_id, paragraph, mark])
            
        # Provide feedback to the admin
        messages.success(request, 'Paragraph and mark stored successfully.')
    return render(request,'keylog_app/question.html')

def create_user(request):
    if request.method == 'POST':
        admin_id = request.session.get('admin_id')
        user_id = request.POST.get('username')
        password = request.POST.get('password')

        # Path to your data.csv file
        csv_file_path = 'users.csv'

        # Write the input to the CSV file along with the admin ID
        with open(csv_file_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([user_id, password,admin_id])
            
        # Provide feedback to the admin
        messages.success(request, 'user details stored successfully.')
    return render(request,'keylog_app/create_user.html')