epoch_loss = 0
    for batch in dataloader:
        imgs = batch['rgb']
        true_masks = batch['truth']

        # Move the images and truth masks to the proper device (cpu or gpu)
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        # Get the model prediction
        masks_pred = model(imgs)

        # Evaluate the loss, which is Cross-Entropy in our case
        loss = criterion(masks_pred, true_masks)
        epoch_loss += loss.item()

        # Update the model parameters
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
