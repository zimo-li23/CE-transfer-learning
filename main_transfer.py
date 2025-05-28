from common import *
from generate import Gen

# generate points randomly

# generate samples without and with Mn
points_others = [[] for i in range(3)]
points_Mn = [[] for i in range(4)]

# Alloy = 4+5 or Total
is_qui = True
# TL or DA
is_TL = True

t = 10
for a in range(0, t):
    for b in np.arange(0, t - a + 1):
        if b >= t: continue
        for c in np.arange(0, t - a - b + 1):
            if c >= t: continue
            for d in np.arange(0, t - a - b - c + 1):
                if d >= t: continue            
                e = t - a - b - c - d
                if e >= t or e < 0: continue
                z = 0
                point = [a, b, c, d, e, 0]
                for (i, n) in enumerate(point[:5]):
                    if n == 0: 
                        z += 1
                    else:
                        point[i] *= 160 // t
                if is_qui:
                    if z > 1: continue
                if e == 0:
                    points_others[3 - z].append(point)
                else:
                    points_Mn[3 - z].append(point)
atom_nums = []
for points in points_others:
    for point in points:
        atom_nums.append(point)
for points in points_Mn:
    for point_Mn in points:
        point_Pd = np.copy(point_Mn)
        point_Pd[-1] = point_Mn[-2]
        point_Pd[-2] = 0
        atom_nums.append(point_Mn)
        atom_nums.append(point_Pd)

proportion_others = [3 / 7, 1 / 4, 1 / 9, 0]
proportion_Mn     = [2 / 7, 3 / 8, 4 / 9, 1 / 2]

# select N samples for augmentation
def select(N):
    proportions = [0, 0, 97 / 152, 55 / 152] if is_qui else [84 / 364, 128 / 364, 97 / 364, 55 / 364]

    if N == 0: return []
    # num of samples for different num of elements 
    n = [0 for i in range(4)]
    ran = range(4, 6) if is_qui else range(2, 6)
    if is_qui:
        for i in ran:
            if i != 4:
                n[i - 2] = int(np.ceil(proportions[i - 2] * N))
        n[2] = N - n[3]
    else:
        for i in ran:
            if i != 3:
                n[i - 2] = int(np.ceil(proportions[i - 2] * N))
        n[1] = N - n[0] - n[2] - n[3]

    rng = np.random.default_rng(42)
    for i in ran: 
        n_Pd = int(np.ceil(proportion_Mn[i - 2] * n[i - 2]))             
        n_others = 0 if proportion_others[i - 2] == 0 else n[i - 2] - 2 * n_Pd
        if n_others < 0: n_others = 0
        n_Mn = n[i - 2] - n_Pd - n_others

        sample_Mn = rng.choice(points_Mn[i - 2], n_Pd, replace = False)
        sample_Pd = np.copy(sample_Mn)
        for sample in sample_Pd:
            sample[-1] = sample[-2]
            sample[-2] = 0
        if n_Mn < n_Pd:
            sample_Mn = rng.choice(sample_Mn, n_Mn, replace = False)
        if n_others != 0:
            sample_others = rng.choice(points_others[i - 2], n_others, replace = False)
            samples = np.concatenate((sample_others, sample_Mn, sample_Pd))
        else:
            samples = np.concatenate((sample_Mn, sample_Pd))
        if i == 2 and not is_qui:
            atom_nums = samples
        elif i == 4 and is_qui:
            atom_nums = samples
        else:
            atom_nums = np.concatenate((atom_nums, samples))
    return atom_nums

if __name__ == '__main__':
 
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
                qui_train, test_temp = train_test_split(qui_train, train_size = p / 100, random_state = 0)            
            train_index[phase] = np.concatenate((train_index[phase], qui_train))
            # train_index[phase] = qui_train

    alpha = 10 ** -4 if target == 'Ef' else 10 ** -2
    init_seed = 0
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)

    X_train, X_test = {}, {}
    y_train, y_test = {}, {}
    X_source, y_source = {}, {}
    # index_s = {}
    for phase in ['fcc', 'bcc']:            
        if is_qui:
            X_train[phase], X_test[phase] = X[phase][train_index[phase]], X[phase][test_index[phase]]
            y_train[phase], y_test[phase] = y[phase][train_index[phase]], y[phase][test_index[phase]]       
        else:
            i = 1
            train_index[phase], test_index[phase] = index[phase][i][0], index[phase][i][1]
            X_train[phase], X_test[phase] = X[phase][train_index[phase]], X[phase][test_index[phase]]
            y_train[phase], y_test[phase] = y[phase][train_index[phase]], y[phase][test_index[phase]]
        
        if not is_TL: atom_nums = select(75) # select 2N samples
        X_source[phase], y_source[phase] = Gen(train_index[phase], atom_nums, property, phase, alpha)
        X_source[phase], y_source[phase] = np.array(X_source[phase]), np.array(y_source[phase])
        if not is_TL:
            X_source[phase], y_source[phase] = np.concatenate((X_train[phase], X_source[phase])), np.concatenate((y_train[phase], y_source[phase])) 
    
    X_train['fcc'], X_train['bcc'] = np.concatenate((X_train['fcc'], np.tile([1,0], (len(X_train['fcc']), 1))), axis = 1), np.concatenate((X_train['bcc'], np.tile([0,1], (len(X_train['bcc']), 1))), axis = 1)
    X_test['fcc'], X_test['bcc'] = np.concatenate((X_test['fcc'], np.tile([1,0], (len(X_test['fcc']), 1))), axis = 1), np.concatenate((X_test['bcc'], np.tile([0,1], (len(X_test['bcc']), 1))), axis = 1)

    X_train, X_test = np.concatenate(list(X_train.values())), np.concatenate(list(X_test.values()))
    y_train, y_test = np.concatenate(list(y_train.values())), np.concatenate(list(y_test.values()))
    
    X_source['fcc'], X_source['bcc'] = np.concatenate((X_source['fcc'], np.tile([1,0], (len(X_source['fcc']), 1))), axis = 1), \
                                         np.concatenate((X_source['bcc'], np.tile([0,1], (len(X_source['bcc']), 1))), axis = 1)
    X_train_s, y_train_s = np.concatenate(list(X_source.values())), np.concatenate(list(y_source.values()))
    X_train_s, X_test_s, y_train_trans, X_scaler, y_scaler = preprocessing(X_train_s, X_test, y_train_s)

    """source training"""
    if is_TL: print("Source: ")
    is_train = True
    training_data = HEA_Dataset(X_train_s, y_train_trans)
    # Create data loaders.
    batch_size = 16   
    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
    
    model = MLP_source().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)

    train_losses, test_losses = [], []

    max_epoch = 1000 if is_train else 0
    patience = 100
    patience_num = 0
    for epoch in range(1, max_epoch + 1):
        train(train_dataloader, model, loss_fn, optimizer)
        # scheduler.step()
        train_loss = test(X_train_s, y_train_s, model, loss_fn, y_scaler)
        train_losses.append(train_loss)
        test_loss = test(X_test_s, y_test, model, loss_fn, y_scaler)
        test_losses.append(test_loss)
        if epoch == 1:
            torch.save(model.state_dict(), "model_source.pth")
            min_loss = test_loss
        print("Epoch:{0}, train loss:{1}, test loss:{2}"
                    .format(epoch, round(train_loss,4), round(test_loss,4)))
        if epoch > 1:
            if test_loss < min_loss:
                patience_num = 0
                torch.save(model.state_dict(), "model_source.pth")
                min_loss = test_loss
            else:
                patience_num += 1
                if patience_num >= patience:
                    break
    
    if is_train:
        # uncomment this if epoch specified
        # torch.save(model.state_dict(), "model_source.pth")
        pass

    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
     
    model.load_state_dict(torch.load("model_source.pth"))
    if not is_TL:
        print("Train: ", end = '')
        test(X_train_s, y_train_s, model, loss_fn, y_scaler, error = True, plot = if_plot)
        print("Test: ", end = '')
        test(X_test_s, y_test, model, loss_fn, y_scaler, error = True, plot = if_plot)
        exit()

    """target training"""
    print("Target: ")
    is_train = True
    X_train, X_test, y_train_trans, X_scaler, y_scaler = preprocessing(X_train, X_test, y_train)

    training_data = HEA_Dataset(X_train, y_train_trans)
    # Create data loaders.
    batch_size = 16   
    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)

    for name, param in model.named_parameters():
        if 'embedding' in name:
            param.requires_grad = False
    
    lr = 0.05 if target == 'Ef' else 0.1
    optimizer = optim.SGD(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

    train_losses, test_losses = [], []

    max_epoch = 1000 if is_train else 0
    patience = 100
    patience_num = 0
    for epoch in range(1, max_epoch + 1):
        train(train_dataloader, model, loss_fn, optimizer)
        if target == 'Ms': scheduler.step()
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
