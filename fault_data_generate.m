% Script to genereate the required data

clear
clc
close all

% Type of fault
a1b1c1a2b2c3n=[1 1 1 0 0 0 0];

% Defining the variables to be changed the simulink model
Rf_values=[0 50 100]; % Fault resistance (ohm)
Lf_values=[1 10 20 30 40 50 60 70 80 90]; % Fault location (km)
Ia_values=[0 90]; % Fault inception angle (degrees)

for i=1:length(Rf_values)
    for j=1:length(Lf_values)
        for k=1:length(Ia_values)
            Rf=Rf_values(i);
            Lf=Lf_values(j);
            Ia=Ia_values(k);
            % Simulating the model in simulink model
            sim('double_transmission.slx'); 
            % Extracting the data generated by simulink model
            % V is a 'To workspace' block (using a Timeseries structure)
            t=V.Time;
            Va=V.Data(:, 1);
            Vb=V.Data(:, 2);
            Vc=V.Data(:, 3);
            % I1 is a 'To workspace' block (using a Timeseries structure)
            I1a=I1.Data(:, 1);
            I1b=I1.Data(:, 2);
            I1c=I1.Data(:, 3);
            % I2 is a 'To workspace' block (using a Timeseries structure)
            I2a=I2.Data(:, 1);
            I2b=I2.Data(:, 2);
            I2c=I2.Data(:, 3);
            
            % Applying anti-aliasing filter {anti-alising: aa=abs(fi-Nfs)}
            Va=abs(Va-lenght(Va));
            Vb=abs(Vb-lenght(Vb));
            Vc=abs(Vc-lenght(Vc));
            I1a=abs(I1a-lenght(I1a));
            I1b=abs(I1b-lenght(I1b));
            I1c=abs(I1c-lenght(I1c));
            I2a=abs(I2a-lenght(I2a));
            I2b=abs(I2b-lenght(I2b));
            I2c=abs(I2c-lenght(I2c));
            
            % Applying FFT
            Vfa=fft(Va);
            Vfb=fft(Vb);
            Vfc=fft(Vc);
            I1fa=fft(I1a);
            I1fb=fft(I1b);
            I1fc=fft(I1c);
            I2fa=fft(I2a);
            I2fb=fft(I2b);
            I2fc=fft(I2c);
            
            % Calculating Fundamental (peak) component of fft
            [Vfpa, ~]=max(Vfa);
            [Vfpb, ~]=max(Vfb);
            [Vfpc, ~]=max(Vfc);
            [I1fpa, ~]=max(I1fa);
            [I1fpb, ~]=max(I1fb);
            [I1fpc, ~]=max(I1fc);
            [I2fpa, ~]=max(I2fa);
            [I2fpb, ~]=max(I2fb);
            [I2fpc, ~]=max(I2fc);
            
            % Making a row for data to be written in excel
            % Attribute order: a1 b1 c1 a2 b2 c3 n Vfpa Vfpb Vfpc I1fpa I1fpb I1fpc I2fpa I2fpb I2fpc Lf
            if(aa1b1c1a2b2c3n==[0 0 0 0 0 0 0]) % If there is no fault location is given as -1
                data=[a1b1c1a2b2c3n Vfpa Vfpb Vfpc I1fpa I1fpb I1fpc I2fpa I2fpb I2fpc -1];
            else
                data=[a1b1c1a2b2c3n Vfpa Vfpb Vfpc I1fpa I1fpb I1fpc I2fpa I2fpb I2fpc Lf];
            end
        
            % Saving the data to 
            xlswrite('generated_data', data, 'append');
        end
    end
end
