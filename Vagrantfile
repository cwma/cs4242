# -*- mode: ruby -*-
# vi: set ft=ruby :

# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure(2) do |config|
  # The most common configuration options are documented and commented below.
  # For a complete reference, please see the online documentation at
  # https://docs.vagrantup.com.

  # Every Vagrant development environment requires a box. You can search for
  # boxes at https://atlas.hashicorp.com/search.
  config.vm.box = "phusion/ubuntu-14.04-amd64"

  # Disable automatic box update checking. If you disable this, then
  # boxes will only be checked for updates when the user runs
  # `vagrant box outdated`. This is not recommended.
  # config.vm.box_check_update = false

  # Create a forwarded port mapping which allows access to a specific port
  # within the machine from a port on the host machine. In the example below,
  # accessing "localhost:8080" will access port 80 on the guest machine.
  # config.vm.network "forwarded_port", guest: 80, host: 8080

  # Create a private network, which allows host-only access to the machine
  # using a specific IP.
  config.vm.network "private_network", ip: "172.16.6.26"

  # Create a public network, which generally matched to bridged network.
  # Bridged networks make the machine appear as another physical device on
  # your network.
  # config.vm.network "public_network"

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  config.vm.synced_folder ".", "/home/vagrant/shared/vagrant_data"

  # Provider-specific configuration so you can fine-tune various
  # backing providers for Vagrant. These expose provider-specific options.
  # Example for VirtualBox:
  #
  # config.vm.provider "virtualbox" do |vb|
  #   # Display the VirtualBox GUI when booting the machine
  #   vb.gui = true
  #
  #   # Customize the amount of memory on the VM:
  #   vb.memory = "1024"
  # end
  #
  # View the documentation for the provider you are using for more
  # information on available options.

  # Define a Vagrant Push strategy for pushing to Atlas. Other push strategies
  # such as FTP and Heroku are also available. See the documentation at
  # https://docs.vagrantup.com/v2/push/atlas.html for more information.
  # config.push.define "atlas" do |push|
  #   push.app = "YOUR_ATLAS_USERNAME/YOUR_APPLICATION_NAME"
  # end

  # Enable provisioning with a shell script. Additional provisioners such as
  # Puppet, Chef, Ansible, Salt, and Docker are also available. Please see the
  # documentation for more information about their specific syntax and use.
  # config.vm.provision "shell", inline: <<-SHELL
  #   sudo apt-get update
  #   sudo apt-get install -y apache2
  # SHELL
  config.vm.provision "shell", inline: <<-SHELL
     sudo apt-get update
     sudo apt-get install -y wget
     sudo apt-get install -y curl
     sudo apt-get install -y vim
     sudo apt-get install -y git
     sudo apt-get install -y build-essential
     sudo apt-get install -y libssl-dev libffi-dev autoconf g++
     sudo apt-get install -y python-pip

     # Set Singapore timezone
     sudo ln -s /usr/share/zoneinfo/Asia/Singapore localtime

     #
     # Installs Python pip for python 2 and 3
     #
     sudo wget https://bootstrap.pypa.io/get-pip.py
     sudo python2 get-pip.py
     sudo python3 get-pip.py

     # sudo wget https://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
     # sudo chmod +x Anaconda3-4.1.1-Linux-x86_64.sh
     # sudo ./Anaconda3-4.1.1-Linux-x86_64.sh

     #
     # Install Python Machine Learning: scikit-learn
     #
     sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
     sudo apt-get install -y build-essential python-setuptools python-dev python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base
     sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
     sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
     sudo pip3 install -U scikit-learn
     sudo pip3 install -U spacy
     sudo pip3 install -U nltk
     sudo pip3 install -U numpy
     sudo pip3 install -U Flask
     sudo pip3 install -U requests
     sudo pip3 install -U futures
     sudo pip3 install -U python-dateutil
     sudo pip3 install -U Cython --install-option="--no-cython-compile"

     # sudo python3 -m spacy.en.download all
     # sudo python3 -m textblob.download_corpora
  SHELL
end
