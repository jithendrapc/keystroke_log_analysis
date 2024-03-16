from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
   # path('', views.frontpage, name='frontpage'),
    path('', views.index, name='index'),
    path('user', views.user, name='user'),
    path('admin', views.admin, name='admin'),
    path('home', views.home, name='home'),
    path('predict_datapoint',views.predict_datapoint,name='predict_datapoint'),
    path('signup/', views.signup, name='signup'),
   path('login/login_view/', views.login_view, name='login_view'),
    path('login/', views.login_view, name='login'),
   # path('login/', auth_views.LoginView.as_view(template_name='keylog_app/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_login_view/', views.admin_login_view, name='admin_login_view'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('question/', views.question, name='question'),
    path('create_user/', views.create_user, name='create_user')
]