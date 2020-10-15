from money_model import MoneyModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    all_trust = []
    all_active = []
    all_cooperation = []
    # This runs the model 100 times, each model executing 10 steps.
    for j in range(1):
        # Run the model
        model = MoneyModel(100)
        for i in range(50):
            model.step()

        # Store the results
        for agent in model.schedule.agents:
            all_trust.append(agent.Trustful)
            all_active.append(agent.Active)
            all_cooperation.append(agent.Cooperation)

    print("Done" + str(max(all_trust)))
    plt.hist(all_trust)
    plt.show()
    print(all_active)
