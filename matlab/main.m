V_reset = -0.080; % -80mV
V_e = -0.075; % -75mV
V_th = -0.040; % -40mV
Rm = 10e6; % membrane resistance
tau_m = 10e-3; % membrane time constant


dt = 0.0002;
T = 0:dt:1; % 1 second simulation

Vm(1) = V_reset;
Im = 5e-9;

for t=1:length(T)-1,
    if Vm(t) > V_th,
        Vm(t+1) = V_reset;
    else,
        Vm(t+1) = Vm(t) + dt * ( -(Vm(t) - V_e) + Im * Rm) / tau_m;
    end;
end;

plot(T,Vm,'b-');
xlabel('Time(s)');
ylabel('Voltage (V)');