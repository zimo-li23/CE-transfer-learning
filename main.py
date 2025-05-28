from common import *


if __name__ == '__main__':
    is_train = True
    is_qui = True
    data = pd.read_csv("HEA_data/Database.csv")
    target = 'Ef' # Ef or Ms
    property = 'Eform (eV/atom)' if target == 'Ef' else 'Ms (mub/atom)'
    X_train, X_test = {}, {}
    y_train, y_test = {}, {}

    X = {'fcc': features(data, 'fcc')[:, :], 'bcc': features(data, 'bcc')[:, :]}
    y = {'fcc': values(data, property, 'fcc'), 'bcc': values(data, property, 'bcc')}
    
    print(f"Using {device} device")

    if_plot = False
    n_splits = 5
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 0)
    index = {phase: list(kf.split(X[phase])) for phase in ["fcc", "bcc"]}
    MAEs = []
    
    train_index, test_index = {"fcc": [], "bcc": []}, {"fcc": [], "bcc": []}
    p = 100  # 4+5 training data size
    for phase in ["fcc", "bcc"]:
        t = []
        df_phase = data[data['phase'] == phase]
        formulas_phase = [structure.split('_')[0] for structure in df_phase['structures'].to_numpy()]

        for (i, formula) in enumerate(formulas_phase):
            count = Formula(formula).count()
            if len(count) >= 4:
                t.append(i)
                test_index[phase].append(i)
            else:
                train_index[phase].append(i)
        # division of 4+5 test set
        qui_train, qui_test = train_test_split(t, train_size = 0.80, random_state = 0)
        test_index[phase] = qui_test
        if p > 0:                       
            if p < 100: 
                qui_train, m = train_test_split(qui_train, train_size = p / 100, random_state = 0)          
            train_index[phase] = np.concatenate((train_index[phase], qui_train))
            # train_index[phase] = qui_train  # uncomment this if Scratch'

    init_seed = 0
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    for phase in ['fcc', 'bcc']:
        if is_qui:
            X_train[phase], X_test[phase] = X[phase][train_index[phase]], X[phase][test_index[phase]]
            y_train[phase], y_test[phase] = y[phase][train_index[phase]], y[phase][test_index[phase]]       
        else:
            i = 1
            train_index, test_index = index[phase][i][0], index[phase][i][1]
            X_train[phase], X_test[phase] = X[phase][train_index], X[phase][test_index]
            y_train[phase], y_test[phase] = y[phase][train_index], y[phase][test_index]

    X_train['fcc'], X_train['bcc'] = np.concatenate((X_train['fcc'], np.tile([1,0], (len(X_train['fcc']), 1))), axis = 1), np.concatenate((X_train['bcc'], np.tile([0,1], (len(X_train['bcc']), 1))), axis = 1)
    X_test['fcc'], X_test['bcc'] = np.concatenate((X_test['fcc'], np.tile([1,0], (len(X_test['fcc']), 1))), axis = 1), np.concatenate((X_test['bcc'], np.tile([0,1], (len(X_test['bcc']), 1))), axis = 1)

    X_train, X_test = np.concatenate(list(X_train.values())), np.concatenate(list(X_test.values()))
    y_train, y_test = np.concatenate(list(y_train.values())), np.concatenate(list(y_test.values()))
    X_train, X_test, y_train_trans, X_scaler, y_scaler = preprocessing(X_train, X_test, y_train)

    training_data = HEA_Dataset(X_train, y_train_trans)
    # Create data loaders.
    batch_size = 16 
    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
    
    model = MLP().to(device)

    lr = 0.005 if target == 'Ef' else 0.001
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)

    train_losses, test_losses = [], []
    max_epoch = 1000 if is_train else 0
    patience = 100
    patience_num = 0
    for epoch in range(1, max_epoch + 1):
        train(train_dataloader, model, loss_fn, optimizer)
        # scheduler.step()
        train_loss = test(X_train, y_train, model, loss_fn, y_scaler)
        test_loss = test(X_test, y_test, model, loss_fn, y_scaler)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch == 1:
            min_loss = test_loss
        print("Epoch:{0}, train loss:{1}, test loss:{2}"
                    .format(epoch, round(train_loss,4), round(test_loss,4)))
        if epoch > 1:
            if test_loss < min_loss:
                patience_num = 0
                torch.save(model.state_dict(), "model.pth")
                min_loss = test_loss
            else:
                patience_num += 1
                if patience_num >= patience:
                    break
   
    model.load_state_dict(torch.load("model.pth"))
    print("Train: ", end = '')
    test(X_train, y_train, model, loss_fn, y_scaler, error = True, plot = if_plot)
    print("Test: ", end = '')
    MAEs.append(test(X_test, y_test, model, loss_fn, y_scaler, error = True, plot = if_plot))
        
    # print(np.array(MAEs).T, np.mean(MAEs, axis = 0))
